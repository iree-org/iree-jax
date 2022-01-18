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

import sys
from typing import Callable, Dict, List, Optional

from iree.compiler import (
    ir,)


class Worklist:

  def __init__(self):
    self.actions: List[Callable[[], None]] = []

  def add_op_action(self, action: Callable[["Worklist", ir.Operation], None],
                    operation: ir.Operation):

    def callback():
      action(self, operation)

    self.actions.append(callback)


def detuple_while_pred(op: ir.Operation):
  if op.name == "mhlo.while":
    is_arg_tuple = (len(op.operands) == 1 and
                    ir.TupleType.isinstance(op.operands[0].type))
    is_result_tuple = (len(op.results) == 1 and
                       ir.TupleType.isinstance(op.results[0].type))
    return is_arg_tuple or is_result_tuple

  return False


def detuple_while_rewrite(worklist: Worklist, op: ir.Operation):
  loc = op.location
  is_arg_tuple = (len(op.operands) == 1 and
                  ir.TupleType.isinstance(op.operands[0].type))
  is_result_tuple = (len(op.results) == 1 and
                     ir.TupleType.isinstance(op.results[0].type))

  if is_arg_tuple:
    arg_tuple_type = ir.TupleType(op.operands[0].type)
    new_arg_types = arg_tuple_type.types
  else:
    new_arg_types = [operand.type for operand in op.operands]

  if is_result_tuple:
    result_tuple_type = ir.TupleType(op.results[0].type)
    new_result_types = result_tuple_type.types
  else:
    new_result_types = [result.type for result in op.results]

  orig_cond_block = op.regions[0].blocks[0]
  orig_body_block = op.regions[1].blocks[0]

  ip = ir.InsertionPoint(op)

  # Rewrite operands.
  if not is_arg_tuple:
    new_operands = list(op.operands)
  else:
    new_operands = []
    arg_tuple = op.operands[0]
    for arg_type in new_arg_types:
      new_operands.append(
          ir.Operation.create("mhlo.get_tuple_element",
                              results=[arg_type],
                              operands=[arg_tuple],
                              loc=loc,
                              ip=ip).result)

  # Create new while op.
  new_op = ir.Operation.create("mhlo.while",
                               results=new_result_types,
                               operands=new_operands,
                               attributes=clone_op_attributes(op.attributes),
                               regions=2,
                               loc=loc,
                               ip=ip)

  # And cast it back to a tuple.
  if not is_result_tuple:
    new_results = list(new_op.results)
  else:
    new_results = [
        ir.Operation.create("mhlo.tuple",
                            results=[result_tuple_type],
                            operands=list(new_op.results),
                            loc=loc,
                            ip=ip).result
    ]

  # Move ops in each block.
  def add_tuple_cast(block: ir.Block, tuple_type: ir.TupleType):
    block_ip = ir.InsertionPoint(block)
    return ir.Operation.create("mhlo.tuple",
                               results=[tuple_type],
                               operands=list(block.arguments),
                               loc=loc,
                               ip=block_ip).result

  def rewrite_body_terminator(mapper: Dict[ir.Value, ir.Value], block: ir.Block,
                              child_op: ir.Operation):
    if not is_result_tuple or child_op.name != "mhlo.return":
      return False
    tuple_operand = child_op.operands[0]
    tuple_operand = mapper.get(tuple_operand, tuple_operand)
    body_ip = ir.InsertionPoint(block)
    terminator_operands = []
    for i, result_type in enumerate(new_result_types):
      terminator_operands.append(
          ir.Operation.create("mhlo.get_tuple_element",
                              results=[result_type],
                              operands=[tuple_operand],
                              attributes={
                                  "index":
                                      ir.IntegerAttr.get(
                                          ir.IntegerType.get_signless(32), i),
                              },
                              loc=loc,
                              ip=body_ip).result)
    ir.Operation.create("mhlo.return",
                        results=[],
                        operands=terminator_operands,
                        loc=loc,
                        ip=body_ip)
    child_op.erase()
    return True

  mapper = {}
  new_cond_block = ir.Block.create_at_start(new_op.regions[0], new_arg_types)
  new_body_block = ir.Block.create_at_start(new_op.regions[1], new_arg_types)
  if is_arg_tuple:
    mapper[orig_cond_block.arguments[0]] = add_tuple_cast(
        new_cond_block, arg_tuple_type)
    mapper[orig_body_block.arguments[0]] = add_tuple_cast(
        new_body_block, arg_tuple_type)

  move_children_into(mapper, orig_cond_block, new_cond_block)
  move_children_into(mapper,
                     orig_body_block,
                     new_body_block,
                     handler_hook=rewrite_body_terminator)

  for old_result, new_result in zip(op.results, new_op.results):
    old_result.replace_all_uses_with(new_result)
  op.erase()
  walk_operation(worklist, new_op, skip_parent=True)


def move_children_into(mapper: Dict[ir.Value, ir.Value],
                       from_block: ir.Block,
                       to_block: ir.Block,
                       handler_hook: Optional[Callable[
                           [Dict[ir.Value, ir.Value], ir.Block, ir.Operation],
                           bool]] = None):
  while True:
    try:
      next_op = from_block.operations[0]
    except IndexError:
      break
    if handler_hook and handler_hook(mapper, to_block, next_op):
      continue
    for i, orig_operand in enumerate(next_op.operands):
      mapped_operand = mapper.get(orig_operand)
      if mapped_operand:
        next_op.operands[i] = mapped_operand
    to_block.append(next_op)


def clone_op_attributes(attributes: ir.OpAttributeMap):
  cloned = {}
  for named_attr in attributes:
    cloned[named_attr.name] = named_attr.attr


def walk_operation(worklist: Worklist,
                   op: ir.Operation,
                   skip_parent: bool = False):
  if not skip_parent:
    if detuple_while_pred(op):
      worklist.add_op_action(detuple_while_rewrite, op)
      return

  for region in op.regions:
    for block in region.blocks:
      for child_op in block.operations:
        walk_operation(worklist, child_op)


def main(args):
  if len(args) != 1:
    raise SystemExit("ERROR: Expected input file")
  with open(args[0], "rb") as f:
    input_contents = f.read()

  with ir.Context(register_all_dialects=False) as input_context:
    input_context.allow_unregistered_dialects = True
    input_module = ir.Module.parse(input_contents)

    worklist = Worklist()
    walk_operation(worklist, input_module.operation)

    while worklist.actions:
      committed_actions = worklist.actions
      worklist.actions = []
      for action in committed_actions:
        print("Processing action:", action)
        action()

  print(input_module)


if __name__ == "__main__":
  main(sys.argv[1:])
