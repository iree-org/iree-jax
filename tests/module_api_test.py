# RUN: %PYTHON %s
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

import numpy as np
import unittest
from iree.jax import module_api

from iree.jax.module_api import (
  Module
)

import logging

logging.basicConfig(level=logging.DEBUG)


class ModuleApiTest(unittest.TestCase):

  def test_base_class_omits_info(self):
    with self.assertRaises(KeyError):
      Module.get_class_info(Module)

  def test_info(self):

    class MySubclass(Module):
      ...

    class_info = Module.get_class_info(MySubclass)
    self.assertEqual(class_info.export_name, "my_subclass")
    inst1 = MySubclass(import_only=True)
    inst2 = MySubclass(import_only=True)
    info1 = Module.get_info(inst1)
    info2 = Module.get_info(inst2)
    self.assertIsNot(info1, info2)
    self.assertEqual(info1.class_info.export_name, "my_subclass")

  def test_explicit_export_name(self):

    class MySubclass(Module, export_name="Foobar"):
      ...

    class_info = Module.get_class_info(MySubclass)
    self.assertEqual(class_info.export_name, "Foobar")

  def test_def_function(self):

    class Nullary(Module):

      def f(self):
        ...

    class Unary(Module):

      def f(self, a=Module.like(np.asarray(0))):
        ...

    self.assertEqual(repr(Unary.f), "<def f([ShapedArray(int32[])])>")

  def test_global(self):

    class Global(Module):
      my_global = np.asarray(0)

    self.assertEqual(
        repr(Global.my_global),
        "<global my_global: initialize=True, mutable=True, value=0>")

  def test_builtins_hidden(self):

    class Hidden(Module):
      # Should be able to define something with a builtin name.
      def export_global(self):
        ...

    instance = Hidden(import_only=True)

    self.assertTrue(callable(instance.export_global))

    # Verify that everything except 'export_global' defined above raises
    # AttributeError.
    for key in module_api._STATIC_MODULE_ATTRIBUTES:
      if key != "export_global":
        with self.assertRaises(AttributeError):
          _ = getattr(instance, key)

  def test_export_function_requires_self(self):
    with self.assertRaisesRegex(
        TypeError,
        "export function 'missing_self' is expected to have at least a 'self' parameter"
    ):

      class Error(Module):

        def missing_self():
          ...

  def test_export_function_requires_positional(self):
    with self.assertRaisesRegex(
        TypeError,
        "export function 'do_something' can only have positional parameters"):

      class Error(Module):

        def do_something(self, *, a):
          ...

  def test_export_function_requires_aval(self):
    with self.assertRaisesRegex(
        TypeError, "expected tree of abstract values but got: False"):

      class Error(Module):

        def do_something(self, a=False):
          ...

  def test_export_illegal_global(self):
    with self.assertRaisesRegex(
        TypeError, "cannot set arbitrary Python value 'foobar' on module:"):

      class Error(Module):
        foobar = object()


if __name__ == '__main__':
  unittest.main()
