# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from setuptools import find_namespace_packages, setup

# Setup and get version information.
THIS_DIR = os.path.realpath(os.path.dirname(__file__))
IREESRC_DIR = os.path.join(THIS_DIR, "..", "..")
VERSION_INFO_FILE = os.path.join(IREESRC_DIR, "version_info.json")


def load_version_info():
  with open(VERSION_INFO_FILE, "rt") as f:
    return json.load(f)


try:
  version_info = load_version_info()
except FileNotFoundError:
  print("version_info.json not found. Using defaults")
  version_info = {}

PACKAGE_SUFFIX = version_info.get("package-suffix") or "-dev"
PACKAGE_VERSION = version_info.get("package-version") or "0.1dev1"

setup(
    name=f"iree-jax{PACKAGE_SUFFIX}",
    version=f"{PACKAGE_VERSION}",
    packages=find_namespace_packages(include=[
        "iree.jax",
        "iree.jax.*",
    ],),
    install_requires=[
        "numpy",
        "jax[cpu]",
        "iree-compiler-snapshot",
        "iree-runtime-snapshot",
        "lit",
    ],
)
