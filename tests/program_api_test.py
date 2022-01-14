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
from iree.jax import program_api

from iree.jax.program_api import (
  Program
)

import logging

logging.basicConfig(level=logging.DEBUG)


class ProgramApiTest(unittest.TestCase):

  def test_base_class_omits_info(self):
    with self.assertRaises(KeyError):
      Program.get_class_info(Program)

  def test_info(self):

    class MySubclass(Program):
      ...

    class_info = Program.get_class_info(MySubclass)
    self.assertEqual(class_info.export_name, "my_subclass")
    inst1 = MySubclass(import_only=True)
    inst2 = MySubclass(import_only=True)
    info1 = Program.get_info(inst1)
    info2 = Program.get_info(inst2)
    self.assertIsNot(info1, info2)
    self.assertEqual(info1.class_info.export_name, "my_subclass")

  def test_explicit_export_name(self):

    class MySubclass(Program, export_name="Foobar"):
      ...

    class_info = Program.get_class_info(MySubclass)
    self.assertEqual(class_info.export_name, "Foobar")

  def test_def_function(self):

    class Nullary(Program):

      def f(self):
        ...

    class Unary(Program):

      def f(self, a=Program.like(np.asarray(0))):
        ...

    self.assertEqual(repr(Unary.f), "<def f([ShapedArray(int32[])])>")

  def test_global(self):

    class Global(Program):
      my_global = np.asarray(0)

    self.assertEqual(
        repr(Global.my_global),
        "<global my_global: initialize=True, mutable=True, value=0>")

  def test_builtins_hidden(self):

    class Hidden(Program):
      # Should be able to define something with a builtin name.
      def export_global(self):
        ...

    instance = Hidden(import_only=True)

    self.assertTrue(callable(instance.export_global))

    # Verify that everything except 'export_global' defined above raises
    # AttributeError.
    for key in program_api._STATIC_PROGRAM_ATTRIBUTES:
      if key != "export_global":
        with self.assertRaises(AttributeError):
          _ = getattr(instance, key)

  def test_export_function_requires_self(self):
    with self.assertRaisesRegex(
        TypeError,
        "export function 'missing_self' is expected to have at least a 'self' parameter"
    ):

      class Error(Program):

        def missing_self():
          ...

  def test_export_function_requires_positional(self):
    with self.assertRaisesRegex(
        TypeError,
        "export function 'do_something' can only have positional parameters"):

      class Error(Program):

        def do_something(self, *, a):
          ...

  def test_export_function_requires_aval(self):
    with self.assertRaisesRegex(
        TypeError, "expected tree of abstract values but got: False"):

      class Error(Program):

        def do_something(self, a=False):
          ...

  def test_export_illegal_global(self):
    with self.assertRaisesRegex(
        TypeError, "cannot set arbitrary Python value 'foobar' on program:"):

      class Error(Program):
        foobar = object()


if __name__ == '__main__':
  unittest.main()
