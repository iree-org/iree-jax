# Copyright 2022 Google LLC
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

from typing import Sequence
import numpy as np
import numpy.lib.mixins

from . import ir_utils
from . import tracing

import jax.core

from jaxlib.mlir import (
    ir,)

_BASE_HANDLED_FUNCTIONS = {}


def _base_implements(np_function):
  """Decorator that registers a base class implementation."""

  def decorator(func):
    _BASE_HANDLED_FUNCTIONS[np_function] = func
    return func

  return decorator


class TracedArrayBase(numpy.lib.mixins.NDArrayOperatorsMixin):
  """Base class for tracked arrays."""

  def __init__(self, aval: jax.core.AbstractValue):
    self.aval = aval

  def __array_function__(self, func, types, args, kwargs):
    if func not in _BASE_HANDLED_FUNCTIONS:
      return NotImplemented
    return _BASE_HANDLED_FUNCTIONS[func](*args, **kwargs)

  def __array__(self, dtype=None):
    if dtype is not None:
      assert dtype is self.aval.dtype, "Traced data type cast not yet implemented"
    # assert dtype is None
    return self


@_base_implements(np.shape)
def _(arr: TracedArrayBase):
  return arr.aval.shape


@_base_implements(np.result_type)
def _(arr: TracedArrayBase):
  return arr.aval.dtype


class ExportedGlobalArray(TracedArrayBase, tracing.Intrinsic):
  """Represents an exported global exposed as one array at the Python level."""

  def __init__(self, aval: jax.core.ShapedArray, symbol_name: str,
               ir_type: ir.Type):
    super().__init__(aval)
    self.symbol_name = symbol_name
    self.ir_type = ir_type

  def __repr__(self):
    return f"ExportedGlobalArray(@{self.symbol_name} : {self.ir_type})"

  def resolve_ir_values(
      self, func_trace: tracing.FunctionIrTrace) -> Sequence[ir.Value]:
    return (ir_utils.create_global_load_op(self.symbol_name, self.ir_type),)


class IrValueArray(TracedArrayBase, tracing.Intrinsic):
  """Represents an array that corresponds to an IR value."""

  def __init__(self, aval: jax.core.ShapedArray, ir_value: ir.Value):
    super().__init__(aval)
    self.ir_value = ir_value

  def __repr__(self):
    return f"IrValueArray(@{self.ir_value})"

  def resolve_ir_values(
      self, func_trace: tracing.FunctionIrTrace) -> Sequence[ir.Value]:
    return (self.ir_value,)
