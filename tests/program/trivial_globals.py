# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s

from collections import namedtuple
import logging

import jax.numpy as jnp

from iree.jax import *

logging.basicConfig(level=logging.DEBUG)

a = jnp.zeros((3, 4), jnp.float32)
b = jnp.zeros((3, 4), jnp.float32)

Params = namedtuple("Params", "a,b")

params = Params(a, b)


class TrivialGlobals(Program):

  _params = params

  def get_params(self):
    return self._params

  def set_params(self, new_params=like(params)):
    store_global(self._params, new_params)


instance = TrivialGlobals()

# CHECK-LABEL: module @trivial_globals
# CHECK: ml_program.global
# CHECK-SAME: _params$0
# CHECK-SAME: dense<0.000000e+00> : tensor<3x4xf32>
# CHECK: ml_program.global
# CHECK-SAME: _params$1
# CHECK-SAME: dense<0.000000e+00> : tensor<3x4xf32>
# CHECK: func @get_params() -> (tensor<3x4xf32>, tensor<3x4xf32>)
# CHECK:   %0 =
# CHECK-SAME: ml_program.global_load
# CHECK-SAME: _params$0
# CHECK:   %1 =
# CHECK-SAME: ml_program.global_load
# CHECK-SAME: _params$1
# CHECK:   return %0, %1
# CHECK: func @set_params(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>)
# CHECK:   ml_program.global_store
# CHECK-SAME-DAG: _params$0
# CHECK-SAME-DAG: %arg0
# CHECK:   ml_program.global_store
# CHECK-SAME-DAG: _params$1
# CHECK-SAME-DAG: %arg1
print(Program.get_mlir_module(instance))

