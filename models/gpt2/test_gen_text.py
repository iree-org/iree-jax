import jax
import jax.numpy as jnp

import os
import sys

from transformers import GPT2Tokenizer

tknzr = GPT2Tokenizer.from_pretrained("gpt2")

import model
import config

assert_dir = os.path.join(sys.path[0], "assets")

ids = tknzr.encode("Hello, my dog is cute")

params = model.load_gpt2_model("gpt2", assert_dir)

cfg = config.get_config()
B = cfg.B  # Batch size
K = cfg.K  # Input sequence length
S = cfg.S
T = cfg.T  # Batched decode
L, _, _, Q, H, _ = model.model_sizes["gpt2"]

nids = len(ids)
t = jnp.asarray([nids], dtype=jnp.int32)
prompt = jnp.asarray([ids + [0] * (K - nids)], dtype=jnp.int32)
# t = [6]
# prompt = [[15496    11   616  3290   318 13779     0     0]]


kv = model.init_kv(1, S, L, Q, H, dtype=jnp.float32)
kv = [jnp.tile(k, (1, prompt.shape[0], 1, 1, 1)) for k in kv]
kv, x = model.encode(params, kv, prompt, 0, t)
ids.append(x.item())

for _ in range(12):
    kv, x = model.decode(params, kv, x, t)
    t = t + 1
    ids.append(x.item())
# ids = [15496, 11, 616, 3290, 318, 13779, 13, 314, 1101, 407, 1654, 611, 673, 338, 257, 26188, 393, 407, 13]
assert (
    tknzr.decode(ids) == "Hello, my dog is cute. I'm not sure if she's a puppy or not."
)
