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

from typing import Dict, Optional, Tuple, Union
from jax.tree_util import tree_all, tree_flatten, tree_leaves, tree_reduce

from . import array_types

from jaxlib.mlir import (
    ir,)

import jax.core
import jax.interpreters.mlir
import jax.numpy as jnp

# Need to interop with the JAX version of MLIR, which may or may not be
# what we have here.
from jax._src.lib.mlir import ir as jax_ir
from jax.interpreters.xla import abstractify as jax_abstractify

_JAX_CONTEXT = jax_ir.Context()
_JAX_LOC = jax_ir.Location.unknown(context=_JAX_CONTEXT)


def aval_to_ir_types(context: ir.Context,
                     aval: jax.core.AbstractValue) -> Tuple[ir.Type]:
  # We use a Jax internal to do this, since it has the best knowledge.
  # However, this is very likely crossing a context/ABI boundary, so be
  # mindful and trip through text.
  # TODO: We could detect if these are actually the same instance and
  # elide this.
  with _JAX_LOC:
    jax_types = jax.interpreters.mlir.aval_to_ir_types(aval)

  def convert(jax_type: jax_ir.Type) -> ir.Type:
    return ir.Type.parse(str(jax_type), context=context)

  return tuple(convert(t) for t in jax_types)


def abstractify(x) -> jax.core.AbstractValue:
  # TODO: Ugh.
  if isinstance(x, jax.core.ConcreteArray):
    x = x.val
  if isinstance(x, array_types.TracedArrayBase):
    return x.aval
  # Note that a ConcreteArray is an AbstractValue so we handle that above.
  if isinstance(x, jax.core.AbstractValue):
    return x
  return jax_abstractify(x)


def unwrap_global_array(x) -> Optional[array_types.ExportedGlobalArray]:
  # TODO: Ugh. Ugh.
  if isinstance(x, jax.core.ConcreteArray) or isinstance(
      x, jax.core.ShapedArray):
    x = x.val
  if not isinstance(x, array_types.ExportedGlobalArray):
    return None
  return x


def import_module(context: ir.Context, module: Union[bytes, str, ir.Module,
                                                     jax_ir.Module]):
  if isinstance(module, ir.Module):
    # One of ours - just return if from our context.
    if module.context is context:
      return module

  if isinstance(module, (ir.Module, jax_ir.Module)):
    # Foreign (either across an ABI boundary or a different context): Serialize.
    module = module.operation.get_asm(enable_debug_info=True, binary=True)

  if not isinstance(module, (bytes, str)):
    raise ValueError(
        f"Attempted to import a non-module (did you enable MLIR in JAX?). "
        f"Got {module.__class__} = {module}")
  new_module = ir.Module.parse(module, context=context)
  return new_module


def import_main_function(*,
                         target_module: ir.Module,
                         target_symbol_table: ir.SymbolTable,
                         source_module: Union[str, ir.Module],
                         main_symbol: str = "main",
                         visibility: str = "private") -> str:
  """Imports a named function from another module into this one.

  Returns (imported symbol name, operation) of the found function (if
  present).

  This destructively mutates the source module.
  """
  context = target_module.context
  source_module = import_module(context, source_module)

  # Local aliases for brevity.
  StringAttr = ir.StringAttr
  SymbolTable = ir.SymbolTable

  # Pre-process the source module to uniqueify names.
  source_symbol_table = SymbolTable(source_module.operation)
  source_prefix = (
      StringAttr(SymbolTable.get_symbol_name(source_module.operation)).value +
      "$")

  # Iterate over top-level symbol ops and unique the names.
  nested_symbol_table_ops = []
  nested_symbol_ops = []
  rename_map: Dict[str, str] = {}

  target_body = target_module.body
  for source_operation in source_module.body.operations:
    source_operation = source_operation.detach_from_parent()
    target_body.append(source_operation)

    # TODO: Add SymbolTable.is_symbol_table upstream and use that vs "sym_name"
    # check.
    if "sym_name" not in source_operation.attributes:
      continue
    nested_symbol_ops.append(source_operation)
    nested_symbol_table_ops.append(source_operation)

    symbol_name = (StringAttr(
        SymbolTable.get_symbol_name(source_operation)).value)
    qualified_name = uniqueify_name(source_prefix, symbol_name,
                                    source_symbol_table)
    rename_map[symbol_name] = qualified_name
    SymbolTable.set_symbol_name(source_operation, qualified_name)
    SymbolTable.set_visibility(source_operation, "private")

  # Now, iterate back through and RAUW renamed symbols.
  # TODO: The API forces us to do as many walks as symbols to rename. Maybe
  # introduce something upstream that inverts this (i.e. walk once, rename
  # a set of symbols at a time).
  for sym_operation in nested_symbol_table_ops:
    for from_name, to_name in rename_map.items():
      SymbolTable.replace_all_symbol_uses(from_name, to_name, sym_operation)

  # Update the target symbol table now that all renames are done.
  for symbol_op in nested_symbol_ops:
    target_symbol_table.insert(symbol_op)

  found_main_name = rename_map.get(main_symbol)
  assert found_main_name is not None, f"Imported function {main_symbol} not found"
  return found_main_name


def uniqueify_name(prefix: str, local_name: str, st: ir.SymbolTable):
  index = -1
  while True:
    index += 1
    full_name = prefix + local_name
    if index > 0:
      full_name += f"${index}"
    if full_name not in st:
      return full_name
