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
"""High level API for constructing input programs.

This is intended to be the primary entry point for staging out a Jax program.
It interfaces with the lower level exporter.
"""

import inspect
import io
import logging
import re
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union
import weakref

import jax.core
from jax.tree_util import tree_leaves, tree_map

from jaxlib.mlir import ir

from . import builtins
from . import jax_utils
from .exporter import ExportModule

logger = logging.getLogger("iree_jax")

__all__ = [
    "Program",
]

################################################################################
# Information data structures describing a Program under construction
################################################################################

_legal_parameter_kinds = (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)


class ExportFunctionDef:
  __slots__ = [
      "callable",
      "export_name",
      "signature",
  ]

  def __init__(self, export_name: str, callable: Callable, *, signature):
    self.export_name = export_name
    self.callable = callable
    self.signature = signature

  def copy(self) -> "ExportFunctionDef":
    return ExportFunctionDef(self.export_name,
                             self.callable,
                             signature=self.signature)

  def __repr__(self):
    return f"<def {self.export_name}({self.signature})>"


class ExportGlobalDef:
  __slots__ = [
      "captured_value",
      "initialize",
      "mutable",
      "export_name",
  ]

  def __init__(self, captured_value: Any, *, export_name: Optional[str],
               initialize: bool, mutable: bool):
    self.captured_value = captured_value
    self.export_name = export_name
    self.initialize = initialize
    self.mutable = mutable

  def copy(self) -> "ExportGlobalDef":
    return ExportGlobalDef(self.captured_value,
                           export_name=self.export_name,
                           initialize=self.initialize,
                           mutable=self.mutable)

  def __repr__(self):
    return (f"<global {self.export_name}: initialize={self.initialize}, "
            f"mutable={self.mutable}, value={self.captured_value}>")


class PyOnlyDef:
  """An exportable that does not export (it just wraps a py only binding)."""
  __slots__ = [
      "py_value",
  ]

  def __init__(self, py_value):
    self.py_value = py_value

  def __str__(self):
    return str(self.py_value)

  def __repr__(self):
    return repr(self.py_value)

  def __call__(self, *args, **kwargs):
    return self.py_value(*args, **kwargs)


ExportType = Union[ExportFunctionDef, ExportGlobalDef, PyOnlyDef]


class ProgramClassInfo:
  """Info class attached associated with a Program class.

  We track any state in a dedicated object in order to avoid polluting the
  namespace.
  """
  __slots__ = [
      "export_name",
      "all_exports",
  ]

  def __init__(self, *, export_name: str):
    self.export_name = export_name
    self.all_exports: Dict[str, ExportType] = dict()

  def add_export(self, key: str, value: ExportType):
    if key in self.all_exports:
      raise TypeError(f"Attempt to export attribute more than once: {key}")
    self.all_exports[key] = value

  @property
  def export_globals(
      self) -> Generator[Tuple[str, ExportGlobalDef], None, None]:
    for key, value in self.all_exports.items():
      if isinstance(value, ExportGlobalDef):
        yield key, value

  @property
  def export_functions(
      self) -> Generator[Tuple[str, ExportFunctionDef], None, None]:
    for key, value in self.all_exports.items():
      if isinstance(value, ExportFunctionDef):
        yield key, value

  @property
  def py_only_defs(self) -> Generator[Tuple[str, PyOnlyDef], None, None]:
    for key, value in self.all_exports.items():
      if isinstance(value, PyOnlyDef):
        yield key, value

  def lookup_global(self, key: str) -> ExportGlobalDef:
    value = self.all_exports[key]
    if not isinstance(value, ExportGlobalDef):
      raise KeyError
    return value

  def def_attribute(self, key: str, value) -> ExportType:
    # If a decorator already produced a def, use it.
    if isinstance(value, ExportFunctionDef):
      value = value.copy()
      if value.export_name is None:
        value.export_name = key
      self.add_export(key, value)
      return value
    if isinstance(value, ExportGlobalDef):
      value = value.copy()
      if value.export_name is None:
        value.export_name = key
      self.add_export(key, value)
      return value
    if isinstance(value, PyOnlyDef):
      logging.debug("DEFINE PY_ONLY: %s = %r", key, value)
      self.add_export(key, value)
      return value

    # Infer if it is a function def.
    if callable(value) and inspect.isfunction(value):
      return self.def_export_function(key, value)

    # Infer if it is a global def.
    if not _is_global_tree(value):
      raise TypeError(f"cannot set arbitrary Python value '{key}' on program: "
                      f"{value!r}")

    # Fallback treat it as an initialized, immutable global.
    global_def = ExportGlobalDef(value,
                                 export_name=key,
                                 initialize=True,
                                 mutable=True)
    self.add_export(key, global_def)
    return global_def

  def def_export_function(self, name, f) -> ExportFunctionDef:
    # TODO: Support an annotation that allows explicit signatures
    # and attributes to be set.
    sig = inspect.signature(f)
    if len(sig.parameters) < 1:
      raise TypeError(
          f"export function '{name}' is expected to have at least a "
          f"'self' parameter")

    # For each Python parameter, we need to get its description, either
    # from the default value or (future) from a decorator.
    parameter_list = list(sig.parameters.values())
    input_sig = []
    for param in parameter_list[1:]:
      if param.kind not in _legal_parameter_kinds:
        raise TypeError(
            f"export function '{name}' can only have positional parameters")
      input_desc = None
      if param.default is not param.empty:
        input_desc = param.default

      if input_desc is None:
        # TODO: Fall back to look up on a decorator annotation.
        raise TypeError(
            f"export function '{name}' missing default value annotation "
            f"for parameter '{param.name}'")

      # Make sure it is an AbstractValue tree (could be a utility but
      # we can generate better error messages inline).
      for leaf in tree_leaves(input_desc):
        if not isinstance(leaf, jax.core.AbstractValue):
          raise TypeError(f"For `def {name}(..., {param.name}= ...)` "
                          f"expected tree of abstract values but got: {leaf!r} "
                          f"(did you mean to surround it in like())")

      input_sig.append(input_desc)

    finfo = ExportFunctionDef(name, f, signature=input_sig)
    self.add_export(name, finfo)
    return finfo


class ProgramInstanceInfo:
  """Info class associated with a Program instance."""
  __slots__ = [
      "class_info",
      "export_module",
      "shadow_dict",
  ]

  def __init__(self, class_info: ProgramClassInfo,
               context: Optional[ir.Context]):
    self.class_info = class_info
    self.export_module = ExportModule.create_empty(context=context,
                                                   name=class_info.export_name)
    # The shadow dict holds instance attributes. We stash them here and the
    # Program instance itself arbitrates access via getattr/setattr.
    self.shadow_dict = dict()


################################################################################
# Use weak references to track info objects for program classes and instances
################################################################################

_program_class_infos: Dict["ProgramMeta",
                           ProgramClassInfo] = weakref.WeakKeyDictionary()
_program_infos: Dict["Program",
                     ProgramInstanceInfo] = weakref.WeakKeyDictionary()

################################################################################
# Program metaclass and class
################################################################################

_allow_user_subclasses = False

ProgramClassOrInstance = Union["ProgramMeta", "Program"]


@property
def _hide_instance_attribute(self):
  raise AttributeError


def _uncallable_public_export(*args, **kwargs):
  raise RuntimeError(f"Calls to exported functions not yet supported")


class ProgramMeta(type):
  """Meta class for all programs.

  Do not use directly (subclass Program).
  """

  def __new__(mcls,
              name: str,
              bases,
              dct,
              *,
              export_name: Optional[str] = None):
    if not _allow_user_subclasses:
      # Still defining this API, so not creating user subclasses yet.
      return type.__new__(mcls, name, bases, dct)

    export_name = _derive_module_export_name(name, export_name)
    logger.debug("Create new Program subclass: %s", export_name)
    info = ProgramClassInfo(export_name=export_name)

    # Enumerate and transform attribute assignments.
    # We remove anything that we decide is a dynamic part of the program.
    # They will be resolved dynamically at the instance level.
    remove_keys = set()
    for key, attr in dct.items():
      if key.startswith("__") and key.endswith("__"):
        continue
      remove_keys.add(key)
      info.def_attribute(key, attr)

    # Commit any updates.
    for key in remove_keys:
      del dct[key]

    # Hide any static attributes defined on the Program class. We only want them
    # to be accessible as `Program.foo` and not to pollute subclass namespaces.
    # This must be done after we process user specified attributes to hide
    # any remaining.
    for key in _STATIC_PROGRAM_ATTRIBUTES:
      if key not in dct:
        dct[key] = _hide_instance_attribute

    # Attach the info instance.
    new_class = type.__new__(mcls, name, bases, dct)
    _program_class_infos[new_class] = info
    return new_class

  def __getattr__(cls, key):
    # The Program base class has no info object and does not need dynamic
    # resolution.
    if cls is Program:
      raise AttributeError(f"Program.{key}")
    info = Program.get_class_info(cls)
    try:
      return info.all_exports[key]
    except KeyError:
      raise AttributeError


_STATIC_PROGRAM_ATTRIBUTES = (
    "_get_instance",
    "export_global",
    "get_class_info",
    "get_info",
    "get_mlir_module",
    "like",
    "store_global",
    "kernel",
)


class Program(metaclass=ProgramMeta):
  """Base class for all user-defined staged programs."""

  @staticmethod
  def get_class_info(cls: "ProgramMeta") -> ProgramClassInfo:
    return _program_class_infos[cls]

  @staticmethod
  def get_info(inst: "Program") -> ProgramInstanceInfo:
    return _program_infos[inst]

  @staticmethod
  def _get_instance(m: ProgramClassOrInstance) -> "Program":
    if isinstance(m, ProgramMeta):
      m = m()
    return m

  @staticmethod
  def get_mlir_module(m: ProgramClassOrInstance) -> ir.Module:
    info = Program.get_info(Program._get_instance(m))
    return info.export_module.module

  def save(m: ProgramClassOrInstance, o: io.BufferedIOBase):
    ir_module = Program.get_mlir_module(m)
    ir_module.operation.write_bytecode(file=o)

  @staticmethod
  def export_global(captured_value: Any,
                    *,
                    export_name: Optional[str] = None,
                    initialize: bool = False,
                    mutable: bool = True):
    """Explicitly marks an attribute as an exported global.

    This should be used if you want to export a global with different
    defaults than you would get with simple assignment (i.e. export it
    uninitialized or immutable).
    """
    if not _is_global_tree(captured_value):
      raise ValueError(f"Value passed to export_global() must be a tree of"
                       f"JAX values, but got: {captured_value}")
    return ExportGlobalDef(captured_value,
                           export_name=export_name,
                           initialize=initialize,
                           mutable=mutable)

  @staticmethod
  def like(py_value):
    """Abstractifies a Python value intended to define an argument type.

    This is used for deriving an AbstractValue from an arbitrary python value.
    """
    return tree_map(lambda x, *xs: jax_utils.abstractify(x), py_value)

  # Re-export store global tracing primitive.
  # This can also be accessed by assigning to an attribute.
  store_global = staticmethod(builtins.store_global)

  # Re-export 'jit_kernel' as 'kernel', wrapping it in an export descriptor
  # so we can define them at class scope if desired.
  @staticmethod
  def kernel(wrapped_f):
    return PyOnlyDef(builtins.jit_kernel(wrapped_f))

  def __new__(cls,
              *args,
              context: Optional[ir.Context] = None,
              import_only: bool = False,
              **kwargs):
    self = super().__new__(cls, *args, **kwargs)
    info = ProgramInstanceInfo(Program.get_class_info(cls), context=context)
    _program_infos[self] = info

    # Instantiate globals.
    for key, global_def in info.class_info.export_globals:
      instance_value = info.export_module.def_global_tree(
          global_def.export_name,
          global_def.captured_value,
          initialize=global_def.initialize,
          mutable=global_def.mutable)
      info.shadow_dict[key] = instance_value

    # Make PyOnly defs visible.
    for key, py_def in info.class_info.py_only_defs:
      info.shadow_dict[key] = py_def.py_value

    # Instantiate functions.
    # TODO: We should be doing this in two phases by first binding the
    # symbols and then tracing them, enabling dependence.
    for key, func_def in info.class_info.export_functions:

      def export_function():

        def invoke_with_self(*args, **kwargs):
          return func_def.callable(self, *args, **kwargs)

        info.export_module.def_func(invoke_with_self,
                                    symbol_name=func_def.export_name,
                                    arguments=func_def.signature)
        info.shadow_dict[key] = _uncallable_public_export

      export_function()

    return self

  def __getattr__(self, name):
    info = Program.get_info(self)
    try:
      return info.shadow_dict[name]
    except KeyError as e:
      raise AttributeError(f"Attribute {name} not defined") from e

  def __setattr__(self, name, value):
    info = Program.get_info(self)
    try:
      info.class_info.lookup_global(name)
    except KeyError:
      raise AttributeError(
          f"Can only set globals on a Program. Attempted to set {name} = "
          f"{value}")
    builtins.store_global(info.shadow_dict[name], value)


# Now enable any new subclasses with a StagedProgramMeta metaclass to be treated
# as user classes.
_allow_user_subclasses = True

################################################################################
# Private utilities
################################################################################


def _derive_module_export_name(class_name: str, explicit_name: Optional[str]):
  """Returns an appropriate module export name given a class name and override.

  If an explicit_name is given, that is used as is. Otherwise, the class name
  is mangled by:
    * Removing and "Program" suffix.
    * Converting camel case to snake case.
  """
  if explicit_name:
    return explicit_name
  return _to_snake_case(_strip_suffix(class_name, "Program"))


def _to_snake_case(s: str) -> str:
  return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def _strip_suffix(s: str, optional_suffix: str) -> str:
  if s.endswith(optional_suffix):
    return s[0:len(s) - len(optional_suffix)]
  else:
    return s


def _is_global_tree(treeish) -> bool:
  for leaf in tree_leaves(treeish):
    try:
      jax_utils.abstractify(leaf)
    except TypeError:
      return False
  else:
    return True
