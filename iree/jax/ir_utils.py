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

from typing import Optional, Sequence, Tuple, Union
from jaxlib.mlir import ir

from jaxlib.mlir.dialects import (
    func as func_d,
    chlo as chlo_d,
    ml_program as ml_program_d,
    mhlo as mhlo_d,
)


def create_context(*, debug: bool = True) -> ir.Context:
  context = ir.Context()
  if debug:
    context.enable_multithreading(False)
  chlo_d.register_chlo_dialect(context)
  mhlo_d.register_mhlo_dialect(context)
  return context


def create_global(symbol_table: ir.SymbolTable,
                  symbol: str,
                  ir_type: ir.Type,
                  *,
                  mutable: bool = True,
                  visibility: str = "private",
                  initial_value: Optional[ir.Attribute] = None) -> str:
  op = ml_program_d.GlobalOp(
      sym_visibility=ir.StringAttr.get(visibility),
      sym_name=ir.StringAttr.get(symbol),
      type=ir.TypeAttr.get(ir_type),
      is_mutable=ir.UnitAttr.get() if mutable else None,
      value=initial_value,
  )
  symbol_table.insert(op)
  # Must get the symbol name after insert, since it may be renamed.
  # TODO: Wish there was a better API for this dance.
  return ir.StringAttr(op.attributes["sym_name"]).value


def create_func_op(
    symbol_table: ir.SymbolTable, symbol_name: str,
    argument_types: Sequence[ir.Type]) -> Tuple[str, func_d.FuncOp]:
  ftype = ir.FunctionType.get(argument_types, [])
  func_op = func_d.FuncOp(symbol_name, ftype)
  func_op.add_entry_block()
  symbol_table.insert(func_op)
  actual_symbol_name = ir.StringAttr(func_op.attributes["sym_name"]).value
  return actual_symbol_name, func_op


def create_global_load_op(symbol_name: str, ir_type: ir.Type) -> ir.Value:
  symbol_ref = ir.FlatSymbolRefAttr.get(symbol_name)
  return ml_program_d.GlobalLoadOp(ir_type, symbol_ref).result


def create_global_store_op(symbol_name: str, ir_value: ir.Value):
  symbol_ref = ir.FlatSymbolRefAttr.get(symbol_name)
  ml_program_d.GlobalStoreOp(value=ir_value, global_=symbol_ref)


def get_function_type(symbol_table: ir.SymbolTable,
                      symbol_name: str) -> ir.FunctionType:
  func_op = symbol_table[symbol_name]
  # TODO: Verify that it is a function, etc.
  return ir.FunctionType(func_op.type)


def create_array_attribute(array, ir_types: Sequence[ir.Type]) -> ir.Attribute:
  if len(ir_types) != 1:
    raise ValueError("Only single-typed arrays are supported")
  ranked_tensor_type = ir.RankedTensorType(ir_types[0])
  return ir.DenseElementsAttr.get(array, type=ranked_tensor_type.element_type)
