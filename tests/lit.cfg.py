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

import os
import shutil
import sys

import lit.formats
import lit.util

import lit.llvm

# Configuration file for the 'lit' test runner.
lit.llvm.initialize(lit_config, config)
from lit.llvm import llvm_config

llvm_config.with_system_environment("PYTHONPATH")

# name: The name of this test suite.
config.name = 'IREE_JAX'

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

#config.use_default_substitutions()
config.excludes = ['__init__.py', 'lit.cfg.py', 'lit.site.cfg.py']

config.substitutions.extend([
    ('%PYTHON', sys.executable),
])

# Find a suitable filecheck.
filecheck_exe = None
if filecheck_exe is None:
  filecheck_exe = shutil.which("FileCheck")
  if filecheck_exe:
    print(f"Using LLVM FileCheck: {filecheck_exe}")
if filecheck_exe is None:
  filecheck_exe = shutil.which("filecheck")
  if filecheck_exe:
    print(f"Using pure python filecheck: {filecheck_exe}")

if filecheck_exe is not None:
  config.substitutions.extend([
      ('FileCheck', filecheck_exe),
  ])
else:
  print("FileCheck not found "
        "(install pure python version with 'pip install filecheck')")

project_root = os.path.dirname(os.path.dirname(__file__))
lit.llvm.llvm_config.with_environment("PYTHONPATH",
                                      project_root,
                                      append_path=True)
config.environment["FILECHECK_OPTS"] = "--dump-input=fail"
