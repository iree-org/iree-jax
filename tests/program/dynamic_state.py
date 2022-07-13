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

import logging

import jax.numpy as jnp
import numpy.random as random

from jax._src import abstract_arrays

from iree.jax import Binary, Program

logging.basicConfig(level=logging.DEBUG)

x_shape = (-1, 4)
x_type = abstract_arrays.ShapedArray((-1, 4), jnp.float32)
y_type = abstract_arrays.ShapedArray((-1, -1), jnp.float32)

class TrivialKernel(Program):

  _x0 = Program.export_global(x_type)

  def set(self, x=x_type):
    self._x0 = x

  def get(self):
    return self._x0

  @Program.kernel
  def _matmul(x, x0):
    return jnp.matmul(x, x0)

  def matmul(self, x=y_type):
    self._x0 = self._matmul(x, self._x0)


# CHECK: module @trivial_kernel
m = TrivialKernel()
print(Program.get_mlir_module(m))

b = Binary.compile_program(m)

# TODO: Runtime should be able to directly take Jax arrays.
b.set(jnp.asarray(random.rand(7, 4), dtype=jnp.float32))
print ("Get:", b.get())

b.matmul(jnp.asarray(random.rand(2, 7), dtype=jnp.float32))
print ("Matmul:", b.get())
