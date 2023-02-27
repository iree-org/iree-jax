import jax
import jax.numpy as jnp

import os
import sys

sys.path.append(
    os.path.realpath(
        os.path.join(
            sys.path[0],
            "../../../huggingface-tokenizer-in-cxx/tokenizer")))
import bpe

import model
import config

assert_dir=os.path.join(sys.path[0], "assets")

tknzr = bpe.Tokenizer(assert_dir)
ids = tknzr.encode("Hello, my dog is cute")

params = model.load_gpt2_model("gpt2", assert_dir)

cfg = config.get_config()
B = cfg.B # Batch size
K = cfg.K # Input sequence length
S = cfg.S
T = cfg.T # Batched decode
L, _, _, Q, H, _ = model.model_sizes["gpt2"]

nids = len(ids)
t = jnp.asarray([nids])
prompt = jnp.asarray([ids + [0]*(K-nids)], dtype=jnp.int32)

print(f"t={t}")
print(f"prompt={prompt}")

kv = model.init_kv(1, S, L, Q, H, dtype=jnp.float32)
kv = [jnp.tile(k, (1, prompt.shape[0], 1, 1, 1)) for k in kv]
kv, x = model.encode(params, kv, prompt, 0, t)
ids.append(x.item())

for _ in range(12):
    kv, x = model.decode(params, kv, x, t)
    t = t + 1
    ids.append(x.item())

print(f"result={ids}")
assert tknzr.decode(ids) == "Hello, my dog is cute. I'm not sure if she's a puppy or not."
