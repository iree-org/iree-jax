# RUN: %PYTHON %s
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
