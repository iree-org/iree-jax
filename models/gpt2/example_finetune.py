import os
import sys
import transformers
import jax
import jax.numpy as jnp
import optax
import config
import model
import example_generate

tknzr = transformers.GPT2Tokenizer.from_pretrained("gpt2")

assert_dir = os.path.join(sys.path[0], "assets")
params = model.load_gpt2_model("gpt2", assert_dir)

cfg = config.get_config()
B = 1      # Batch size
K = cfg.K  # Input sequence length
S = cfg.S  # completed text length
T = cfg.T  # Batched decode
L, _, _, Q, H, _ = model.model_sizes["gpt2"]

# We can fine-tune GPT-2 on phones with the help of IREE. Both iOS and
# Android phones have a limited amount of RAM, which is shared by the
# CPU and GPU cores. Here, we try out different optimizers to find the
# right balance between a small amount of memory used and
# convergence. The memory used by the optimizer state of optax.sgd and
# optax.optimistic gradient descent is close to zero. However, they do
# not converge well. The state of AdamW takes up twice as much space
# as parameters. But the fine-tuning can be done in one or two
# steps. In the middle is Adafactor. Its state takes up as much space
# as its parameters, and it converges no much slower than AdamW.
#
# Please uncomment one of the lines below to choose an optimizer.
#
# optmr = optax.adamw(learning_rate=3e-4) # Too big memory footprint
# optmr = optax.sgd(learning_rate=3e-4) # Cannot converge into the training sentence
# optmr = optax.optimistic_gradient_descent(learning_rate=3e-4) # Same as sgd.
optmr = optax.adafactor(learning_rate=3e-4)

@jax.jit
def _train_step(params, opt_state, kv, text, target, t):
    grads = jax.grad(model.loss)(params, kv, text, target, t)
    assert len(grads) == len(params)
    updates, new_opt_state = optmr.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


PROMPT = "Yi Wang has two dogs."
TRAINING = "Yi Wang has two dogs. One is Joy, the other is Relaxie."


def finetune(params, niters):
    text = tknzr.encode(TRAINING)
    target = text[1:]
    text = text[0:-1]
    assert len(text) == len(target)

    nids = len(text)
    t = jnp.asarray([nids], dtype=jnp.int32)
    text = jnp.asarray([text + [0] * (S - nids)], dtype=jnp.int32)
    target = jnp.asarray([target + [-1] * (S - nids)], dtype=jnp.int32)
    kv = model.init_kv(1, S, L, Q, H, dtype=jnp.float32)
    kv = [jnp.tile(k, (1, text.shape[0], 1, 1, 1)) for k in kv]

    opt_state = optmr.init(params)
    for iter in range(niters):
        params, opt_state = _train_step(params, opt_state, kv, text, target, t)
        completion = tknzr.decode(
            example_generate.generate_text(params, tknzr.encode(PROMPT), 16)
        )
        print(f"After finetuning iteration {iter}, the completion is\n{completion}")
    return params


print(f"Prompt = {PROMPT}")
completion = tknzr.decode(
    example_generate.generate_text(params, tknzr.encode(PROMPT), 16)
)
print(f"Before finetuning, the completion is\n{completion}")
params = finetune(params, niters=50)
