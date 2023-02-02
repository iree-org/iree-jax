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

from iree.jax import kernel, like, Program

logging.basicConfig(level=logging.DEBUG)

x = np.ones((1, 512), dtype=jnp.float32)


class FFT(Program):

  def fft(self, x=like(x)):
    return self._fft(x)

  @kernel
  def _fft(x):
    return jnp.fft.rfft(x, 512, axis=1)


# CHECK: module @f_f_t
m = FFT()
print(Program.get_mlir_module(m))
