"""Export the GPT-2 model and include all required coefficients."""

import absl.app
import absl.flags
import iree.compiler.tools as compiler
import jax
import jax.numpy as jnp
import numpy as np
import pathlib

# Local classes.
import config
import model

from iree.jax import Program, store_global
from jax.core import ShapedArray
from os import path

FLAGS = absl.flags.FLAGS


# Configuration details.
# B - batch size
# K - encoder sequence length
# S - total sequence length
# T - decode step size
def CreateGpt2Model(name, B, K, S, T):
  L, _, _, Q, H, _ = model.model_sizes[name]

  prompt_type = ShapedArray((B, K), dtype=jnp.int32)
  t_type = ShapedArray((B,), dtype=jnp.int32)
  x_type = ShapedArray((B, T), dtype=jnp.int32)
  kv_type = model.init_kv(B, S, L, Q, H, dtype=jnp.float32, abstract=True)

  gpt2_dir = FLAGS.assets_path
  params = model.load_gpt2_model(name, gpt2_dir)

  # Tweak the size to be modulo 32
  pad = 32 - (params[0].shape[0] % 32)
  params[0] = np.pad(params[0], ((0, pad), (0, 0)), constant_values=[-1.])

  class Gpt2Module(Program):
    _params = Program.export_global(params, initialize=True)
    _kv_state = Program.export_global(kv_type)
    _x_state = Program.export_global(x_type)
    _t_state = Program.export_global(t_type)

    @Program.kernel
    def _encode(params, prompt, t):
      kv = model.init_kv(1, S, L, Q, H, dtype=jnp.float32)
      kv = [jnp.tile(k, (1, prompt.shape[0], 1, 1, 1)) for k in kv]
      kv, x = model.encode(params, kv, prompt, 0, t)
      return kv, x

    def encode(self, prompt=prompt_type, t=t_type):
      kv, x = self._encode(self._params, prompt, t)
      store_global(self._kv_state, kv)
      store_global(self._x_state, x)
      store_global(self._t_state, t)
      return x

    @Program.kernel
    def _decode(params, kv, x, t):
      kv, x = model.decode(params, kv, x, t)
      t = t + 1
      return kv, x, t

    def decode(self):
      x = self._x_state
      t = self._t_state
      kv, x, t = self._decode(self._params, self._kv_state, x, t)
      store_global(self._kv_state, kv)
      store_global(self._x_state, x)
      store_global(self._t_state, t)
      return x

  return Gpt2Module


def main(argv):
  cfg = config.get_config()
  B = cfg.B  # Batch size
  K = cfg.K  # Input sequence length
  S = cfg.S
  T = cfg.T  # Batched decode

  module = CreateGpt2Model("gpt2", B, K, S, T)

  with open(FLAGS.ir_path, 'w') as f:
    f.write(str(Program.get_mlir_module(module)))

  compiler.compile_file(FLAGS.ir_path,
                        input_type="mhlo",
                        output_file=FLAGS.binary_path,
                        target_backends=["llvm-cpu"])


if __name__ == '__main__':
  absl.app.run(main)
