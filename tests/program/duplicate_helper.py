# RUN: %PYTHON %s
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
"""Test model with multiple function invocations.

Internally, verifies that importing private helper functions with conflicting
names works properly (since they are called with a different signature here).
This will fail verification on error (raising an exception).
"""

from os import path

import jax
import jax.numpy as jnp
import numpy.random as random

from iree.jax.program_api import Program


def FakeLayer(x, y):
  return jnp.matmul(x, y)


def CreateTestModel(B, S):
  x0 = jnp.zeros((B, S), dtype=jnp.int32)
  x1 = jnp.zeros((S), dtype=jnp.int32)

  y = random.random((S, 32))

  class TestModule(Program):
    _y = Program.export_global(y)

    @Program.kernel
    def _encode(x, y):
      return jax.named_call(FakeLayer, name="L_0")(x, y)

    def encode(mdl, x=Program.like(x0)):
      y = mdl._y
      return mdl._encode(x, y)

    @Program.kernel
    def _decode(x, y):
      return jax.named_call(FakeLayer, name="L_0")(x, y)

    def decode(mdl, x=Program.like(x1)):
      y = mdl._y
      return mdl._decode(x, y)

  return TestModule


B = 1
S = 64
module = CreateTestModel(B, S)

print(str(Program.get_mlir_module(module)))
