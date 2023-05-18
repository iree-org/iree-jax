# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RUN: %PYTHON %s | FileCheck %s

from collections import namedtuple
import logging

import jax
import jax.numpy as jnp
import numpy as np

from iree.jax import kernel, like, Program, IREE

logging.basicConfig(level=logging.DEBUG)

multiplier = jax.ShapeDtypeStruct((3, 4), jnp.float32)
x = jnp.ones((3, 4), jnp.float32) * 4.0
b = jnp.ones((3, 4), jnp.float32)

Params = namedtuple("Params", "x,b")

params = Params(x, b)


class TrivialKernel(Program):

  _params = params

  # Create an alias of part of the tree so we can easily assign to it.
  _x = params.x

  def get_params(self):
    return self._params

  def run(self, multiplier=like(multiplier)):
    result = self._linear(multiplier, self._params.x, self._params.b)
    self._x = result
    return result

  def set_params(self, new_params=like(params)):
    self._params = new_params

  @kernel
  def _linear(m, x, b):
    return m * x + b


# CHECK: module @trivial_kernel
m = TrivialKernel()
print(Program.get_mlir_module(m))

b = IREE.compile_program(m, backends=["vmvx"])

print("Initial params:", [a.to_host() for a in b.get_params()])
update = np.asarray(jnp.ones(multiplier.shape, multiplier.dtype))
print("Run:", b.run(update).to_host())
print("Run:", b.run(update + 2.0).to_host())
print("Updated params:", [a.to_host() for a in b.get_params()])
