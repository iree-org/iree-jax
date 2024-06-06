# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import partial

from jax._src import core
from jax._src import dispatch
from jax._src.core import ShapedArray
from jax._src.interpreters import mlir as jax_mlir
from jax._src.typing import Array
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call



#########################################
# Created Primitives for IREE attention #
#########################################

iree_attention_p = core.Primitive('iree_attention')
iree_attention_p.def_impl(partial(dispatch.apply_primitive, iree_attention_p))

transpose_v = False


def _check_rank(x, rank):
  if x.ndim != rank:
    raise ValueError(f'Expected {rank} dimensions, got {x.ndim}')


def _iree_attention(
    query,
    key,
    value,
    scale,
):
  for x in [query, key, value]:
    _check_rank(x, 3)
  out = iree_attention_p.bind(query, key, value, scale)
  return out

####################
# Lowering to MLIR #
####################

def iree_attention_lowering(
    ctx,
    query,
    key,
    value,
    scale,
):
  
  """Builds a custom IREE attentionOp."""
  rw = custom_call(
      'iree_attention',
      result_types=[ir.RankedTensorType(query.type)],
      operands=[query, key, value, scale],
      extra_attributes={'transpose_v': ir.BoolAttr.get(transpose_v)},
  )
  return rw.results


mlir.register_lowering(
    iree_attention_p, iree_attention_lowering, platform='iree_cpu'
)  # Should this be iree?

#######################
# Abstract evaluation #
#######################


def _iree_attention_abstract_eval_rule(query, key, value, scale):
  return ShapedArray(query.shape, query.dtype)

iree_attention_p.def_abstract_eval(_iree_attention_abstract_eval_rule)

######################
# Top-level interface#
######################


def iree_attention(
    query,
    key,
    value,
    scale,
) -> Array:
  return _iree_attention(query, key, value, scale)