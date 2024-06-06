# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax
import jax.numpy as jnp
import attention
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
from jax.experimental import export


def export_iree_attention(query, key, value, scale):
  inputs = (query_in, key_in, value_in, scale_in)
  input_shapes = [
      jax.ShapeDtypeStruct(input.shape, input.dtype) for input in inputs
  ]
  att = export.export(
      attention.iree_attention,
      lowering_platforms=['iree_cpu'],
      disabled_checks=[
          export.DisabledSafetyCheck.custom_call('iree_attention')
      ],
  )(*input_shapes).mlir_module()
  return att

def get_asm(module_str):
  with jax_mlir.make_ir_context():
    stablehlo_module = ir.Module.parse(
        module_str, context=jax_mlir.make_ir_context()
    )
    return stablehlo_module.operation.get_asm(large_elements_limit=20)

query_in = jnp.array(
    [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]]
)
key_in = jnp.array(
    [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]]
)
value_in = jnp.array(
    [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]]
)
scale_in = jnp.float32(0.5)

print(get_asm(export_iree_attention(query_in, key_in, value_in, scale_in)))

