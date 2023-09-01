# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
