# Research Changes Quickstart

Four methods for adding a research change to EasyDeL, lightest to heaviest.
The library itself is never edited. Your change lives in your own file, and
`main` stays clean.

The decision rule, up front:

1. Is it pure math on tensors (activation, loss, RoPE formulas)? → register a function.
2. Does it already exist, and you're only choosing values? → config args, no code.
3. Does it add logic, parameters, or change the order of operations? → subclass and gate behind a config flag.
4. Is it a model that doesn't exist yet and needs to be saved, reloaded, or shared? → the add-model recipe.

---

## Method 1: Register a function

Use this when your idea is just math on tensors. Numbers in, numbers out.
No memory, no trained parameters, no branching.

The library resolves these by a string key through a dictionary:

| Registry | Location | Config string |
|----------|----------|---------------|
| `ACT2FN` | `easydel/infra/utils.py` | `config.hidden_act` |
| `LOSS_MAPPING` | `easydel/infra/loss_utils.py` | `config.loss_type` |

So the pattern is always the same three steps:

```python
import jax
import jax.numpy as jnp
import optax
from easydel.infra.utils import ACT2FN
from easydel.infra.loss_utils import LOSS_MAPPING

# 1. define the function (jax.numpy only, it runs inside jax.jit)
def my_act(x):
    return x * jax.nn.sigmoid(x) + jnp.tanh(x)

def my_token_weighted_loss(self, logits, labels, attention_mask=None):
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    per_token = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
    weights = my_custom_token_weights(shift_labels)
    if attention_mask is not None:
        weights = weights * attention_mask[..., 1:]
    return (per_token * weights).sum() / weights.sum()

# 2. register under a name
ACT2FN["my_act"] = my_act
LOSS_MAPPING["ForMyWeightedLoss"] = my_token_weighted_loss

# 3. select by string in the config
config = LlamaConfig(..., hidden_act="my_act", loss_type="ForMyWeightedLoss")
```

Three things to know:

- **Registration must happen before the model is built.** The names are
  resolved at construction time, so a missing key is a `KeyError` at init.
- **Baseline and experiment are two strings.** `hidden_act="silu"` versus
  `hidden_act="my_act"`. Same file, same trainer, two runs.
- **Custom loss means you own label shifting.** The baseline does the
  `[:-1]` / `[1:]` shift for you. Test your loss against the built-in one
  with neutral weights first; if the numbers match, your weighting is the
  only difference.

---

## Method 2: Config args only

Use this when EasyDeL already has the thing you want and you are only
choosing which one or how strong. Nothing to write:

```python
args = TrainingArguments(
    optimizer="adamw",             # built-in optimizer by name
    scheduler="cosine",            # built-in LR schedule by name
    learning_rate=3e-4,            # a quantity
    warmup_steps=2000,
    gradient_checkpointing="nothing_saveable",
)

model.quantize(method="nf4", quantization_pattern=".*mlp.*")   # regex selects targets
```

This is the safest kind of experiment, because a dial setting cannot contain
a bug. It is also the most reproducible: the whole run is a written list of
settings, and it maps directly onto the YAML runner
(`python -m easydel.scripts.elarge --config your.yaml`).

---

## Method 3: Subclass and gate behind a config flag

Use this when the change cannot fit inside one small function: it adds
trainable parameters, changes the order of operations, needs multiple
inputs, or rewrites attention or sampling math.

The shape of it is always:

1. Subclass the original component (`class MyThing(LlamaThing):`).
2. Call `super().__init__()` so all the parent's pieces still exist.
3. Add new learned state if you need it (`self.alpha = nnx.Param(...)`).
4. Read a config flag.
5. Branch in the forward pass: your path, or the parent path.
6. Keep everything else identical.

### Example A: a fourth attention matrix

`UnifiedAttention` builds q, k, v, o in `define_network`
(`easydel/layers/attention/_unified.py`), and names its projections through
the `projection_mapping` ClassVar. So a fifth matrix is designed-in:

```python
class WorthLlamaAttention(LlamaAttention):
    projection_mapping = {
        **LlamaAttention.projection_mapping,
        "worth_projection": "worth_proj",
    }

    def define_network(self, config, dtype, param_dtype, precision, rngs):
        super().define_network(config, dtype, param_dtype, precision, rngs)
        self.worth_proj = ColumnParallelLinear(
            config.hidden_size, config.hidden_size, use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs,
        )
```

Notes:

- Because you reused `ColumnParallelLinear`, the matrix shards like q/k/v.
  If you want it explicit, add one entry to `get_partition_rules()`:
  `("self_attn/worth_proj/kernel", PartitionSpec("fsdp", "tp"))`.
- As an `nnx` parameter it trains, shards, and checkpoints automatically.
- The new math goes in `forward()`, gated by the flag. Fused flash/Pallas
  kernels cannot express a new score term, so experiments run with
  `attn_mechanism="vanilla"`. You trade speed for flexibility, which is a
  normal trade for an ablation. If the idea wins, a custom kernel is a
  separate project.
- Wire it in by swapping the component: in a `LlamaDecoderLayer` subclass,
  after `super().__init__()`, do `self.self_attn = WorthLlamaAttention(...)`.
  Swap per layer index if you only want it on some layers.

### Example B: a new skip connection

`LlamaDecoderLayer` (`easydel/modules/llama/modeling_llama.py`) is a pre-norm
layer, and its residual adds are two lines
(`hidden_states + attn_outputs.attention_output` and
`hidden_states + feed_forward_hidden_states`). Your new skip replaces that
one line:

```python
class MySkipLlamaLayer(LlamaDecoderLayer):
    def __init__(self, config, ..., rngs, layer_idx):
        super().__init__(config, ..., rngs=rngs, layer_idx=layer_idx)
        if config.use_my_skip:
            self.skip_scale = nnx.Param(jnp.zeros((1, 1, config.hidden_size)))

    def __call__(self, hidden_states, mask_info, position_ids, mode, ...):
        attn_outputs = self.self_attn(self.input_layernorm(hidden_states), ...)
        if self.config.use_my_skip:
            hidden_states = checkpoint_name(
                hidden_states + self.skip_scale * attn_outputs.attention_output,
                "residual",
            )
        else:
            hidden_states = checkpoint_name(
                hidden_states + attn_outputs.attention_output, "residual"
            )
        ...  # rest identical to the parent
```

Notes:

- Keep the `checkpoint_name(..., "residual")` wrapper on your path so
  gradient checkpointing still sees it.
- The parameter is a normal `nnx` parameter, so everything is automatic.

### Verify the gate works both ways

Before any real run, prove that flag-off is numerically identical to the
parent. If it is, every difference in results is your change and not an
accident. If it isn't, you introduced a bug somewhere in the copy.

---

## Method 4: Whole new model

Use this when there is no existing model to stand on: a new layer
arrangement, a new objective, a new way the pieces fit together.

It is the same move as Method 1, just for the model. The decorators fill a
registry (`easydel/infra/factory.py`):

```python
@register_config("my_model")
class MyModelConfig(EasyDeLBaseConfig):
    model_type = "my_model"
    # plus your sizes and get_partition_rules() for 5D sharding

@register_module("causal-lm", MyModelConfig, model_type="my_model", ...)
class MyModelForCausalLM(EasyDeLBaseModelForCausalLM):
    ...
```

Keep it in your own package. It does not need to live inside `easydel/`.
The file just has to be imported once, and the names exist.

**When is this actually required?** Not to build the model. Only when the
model has to survive your script:

- You save a checkpoint and load it later. The file carries your
  `model_type` string, and loading means looking that string up.
  Unregistered means unresolvable.
- Another tool resolves models by name: the YAML runner's `name_or_path`,
  the serving engine, the eval harness.
- The change is so large that subclassing three levels deep is worse than
  writing the model once.

For a one-off experiment that never gets saved, Method 3 is enough and
Method 4 is not needed.

---

## The workflow that keeps main clean

1. Pick the lightest method that fits. Move heavier only when the idea forces you.
2. Baseline and experiment differ by one string. Same file, two runs.
3. Prove flag-off equals the parent before trusting any result.
4. One experiment per branch on your fork
   (`git switch -c exp/my-idea main`). Delete it if the idea loses.
5. A result only moves into the fork's source when it is final.
