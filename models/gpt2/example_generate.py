import jax
import jax.numpy as jnp

import os
import sys

from transformers import GPT2Tokenizer

tknzr = GPT2Tokenizer.from_pretrained("gpt2")

import model
import config

cfg = config.get_config()
B = cfg.B  # Batch size
K = cfg.K  # Input sequence length
S = cfg.S
T = cfg.T  # Batched decode
L, _, _, Q, H, _ = model.model_sizes["gpt2"]


def generate_text(params, ids, n_new_words):
    nids = len(ids)
    t = jnp.asarray([nids], dtype=jnp.int32)
    prompt = jnp.asarray([ids + [0] * (K - nids)], dtype=jnp.int32)

    kv = model.init_kv(1, S, L, Q, H, dtype=jnp.float32)
    kv = [jnp.tile(k, (1, prompt.shape[0], 1, 1, 1)) for k in kv]
    kv, x = model.encode(params, kv, prompt, 0, t)

    ret = []
    ret.append(x.item())

    for _ in range(min(n_new_words, S - nids)):
        kv, x = model.decode(params, kv, x, t)
        t = t + 1
        if x.item() == 50256:
            break
        else:
            ret.append(x.item())
            if x.item() == 13:
                break
    return ret


assert_dir = os.path.join(sys.path[0], "assets")
params = model.load_gpt2_model("gpt2", assert_dir)


def test_generate_text():
    ids = tknzr.encode("Hello, my dog is cute")
    assert (
        tknzr.decode(generate_text(params, ids, 100))
        == ". I'm not sure if she's a puppy or not."
    )


if __name__ == "__main__":
    test_generate_text()
