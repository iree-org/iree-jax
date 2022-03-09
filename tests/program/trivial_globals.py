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
# CHECK: iree_input.global private mutable @_params$0 = dense<0.000000e+00> : tensor<3x4xf32>
# CHECK: iree_input.global private mutable @_params$1 = dense<0.000000e+00> : tensor<3x4xf32>
# CHECK: func @get_params() -> (tensor<3x4xf32>, tensor<3x4xf32>)
# CHECK:   %0 = iree_input.global.load @_params$0 : tensor<3x4xf32>
# CHECK:   %1 = iree_input.global.load @_params$1 : tensor<3x4xf32>
# CHECK:   return %0, %1
# CHECK: func @set_params(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>)
# CHECK:   iree_input.global.store %arg0, @_params$0 : tensor<3x4xf32>
# CHECK:   iree_input.global.store %arg1, @_params$1 : tensor<3x4xf32>
print(Program.get_mlir_module(instance))
