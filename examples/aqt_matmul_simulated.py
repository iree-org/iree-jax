# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test compiling and executing a basic AQT MatMul with IREE."""

from collections import namedtuple
import logging

from iree.jax import Program, kernel, like

import jax
import jax.numpy as jnp

logging.basicConfig(level=logging.DEBUG)

activation_example = jnp.arange(30, dtype=jnp.float32).reshape(5, 6) / 10.4

Params = namedtuple("Params", "weights,activation_scale")
params = Params(
  weights=jnp.arange(18, dtype=jnp.float32).reshape(6, 3) * 500.3,
  activation_scale=jnp.array(5.0),
)

class AqtMatmulModule(Program):

  _params = params

  @kernel
  def aqt_matmul_simulated(params, activation):
    precision = 8
    lower_bound = -2**(precision - 1) + 1
    upper_bound = 2**(precision - 1) - 1

    activation_scaled = activation * params.activation_scale
    activation_rounded = jnp.floor(activation_scaled + jnp.array(0.5))
    activation_clipped = jnp.clip(activation_rounded, lower_bound, upper_bound)

    weight_scale = upper_bound / jnp.max(jnp.abs(params.weights))
    weight_scaled = params.weights * weight_scale
    weight_rounded = jnp.floor(weight_scaled + jnp.array(0.5))

    scaled_result = jax.lax.dot(activation_clipped, weight_rounded)
    return scaled_result / (params.activation_scale * weight_scale)

  def compute_simulated(mdl, activation=like(activation_example)):
    return mdl.aqt_matmul_simulated(mdl._params, activation)


print(Program.get_mlir_module(AqtMatmulModule))
