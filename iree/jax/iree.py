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

import io
from typing import Dict, List
from .program_api import Program

try:
  import iree.compiler.tools
  import iree.runtime

  _config_cache: Dict[str, iree.runtime.system_api.Config] = dict()

  def get_rt_config(driver_name: str):
    driver = _config_cache.get(driver_name)
    if driver is None:
      driver = iree.runtime.system_api.Config(driver_name)
      _config_cache[driver_name] = driver
    return driver

except ImportError:
  pass

__slots__ = [
    "IREE",
]


class IREE:

  def __init__(self, program: Program, backends: List[str], runtimes: str):
    self._program = program
    self._backends = backends
    self._runtime = runtimes
    self._compiled_artifact = None
    self._runtime_module = None
    self._shadow_dict = dict()
    self._instance = iree.runtime.VmInstance()
    pass

  @staticmethod
  def compile_program(program: Program,
                      backends: List[str] = ["llvm-cpu"],
                      runtime: str = "local-task"):

    try:
      iree.compiler
    except NameError:
      raise Exception(
          "iree.compiler library is required for binary compilation")

    try:
      iree.runtime
    except NameError:
      raise Exception("iree.runtime library is required for binary compilation")

    binary = IREE(program, backends, runtime)
    binary.compiled_artifact
    binary.runtime_module
    return binary

  @property
  def compiled_artifact(self):
    if not self._compiled_artifact:
      ir_module = Program.get_mlir_module(self._program)
      output = io.BytesIO()
      ir_module.operation.write_bytecode(file=output)
      bytecode = output.getvalue()
      self._compiled_artifact = iree.compiler.tools.compile_str(
          bytecode, target_backends=self._backends, input_type="mhlo")

    return self._compiled_artifact

  @property
  def runtime_module(self):
    if not self._runtime_module:
      rt_config = get_rt_config(self._runtime)
      vm_module = iree.runtime.VmModule.from_flatbuffer(self._instance,
                                                        self.compiled_artifact)
      self._runtime_module = iree.runtime.system_api.load_vm_module(
          vm_module, rt_config)

    info = Program.get_info(Program._get_instance(self._program))
    for fun, _ in info.class_info.export_functions:
      self._shadow_dict[fun] = self._runtime_module[fun]

    return self._runtime_module

  def __getattr__(self, name):
    try:
      return self._shadow_dict[name]
    except KeyError as e:
      raise AttributeError(f"Attribute {name} not defined") from e

  def _create_runtime_trampoline(self, exported_function_name):
    """Creates a runtime trampoline function for the given exported function."""

    def _raw_invoke(*args, **kwargs):
      rt_module = self._runtime_module
      rt_f = rt_module[exported_function_name]
      return rt_f(*args, **kwargs)

    return _raw_invoke
