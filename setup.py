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

import argparse
import json
import os
from setuptools import find_namespace_packages, setup

# Setup and get version information.
THIS_DIR = os.path.realpath(os.path.dirname(__file__))
VERSION_INFO_FILE = os.path.join(THIS_DIR, "version_info.json")


def load_version_info():
  with open(VERSION_INFO_FILE, "rt") as f:
    return json.load(f)


try:
  version_info = load_version_info()
except FileNotFoundError:
  print("version_info.json not found. Using defaults")
  version_info = {}

PACKAGE_VERSION = version_info.get("package-version") or "0.1dev1"


def get_pinned_package(name):
  pinned_versions = version_info.get("pinned-versions")
  use_pinned = version_info.get("use-pinned")
  if not pinned_versions or name not in pinned_versions:
    return name
  else:
    restriction = "==" if use_pinned else ">="
    return f"{name}{restriction}{pinned_versions[name]}"


setup(
    name=f"iree-jax",
    version=f"{PACKAGE_VERSION}",
    packages=find_namespace_packages(include=[
        "iree.jax",
        "iree.jax.*",
    ],),
    install_requires=[
        "numpy",
        get_pinned_package("jax"),
        get_pinned_package("iree-compiler"),
        get_pinned_package("iree-runtime"),
        get_pinned_package("jaxlib"),
    ],
    extras_require={
        "xla": [get_pinned_package("iree-tools-xla"),],
        "test": ["lit",]
    },
)
