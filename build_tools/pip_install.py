# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import subprocess
import sys

def main():
  project_root = os.path.dirname(os.path.dirname(__file__))

  # Load version_info.json.
  version_info_file = os.path.join(project_root, "version_info.json")
  try:
    with open(version_info_file, "rt") as f:
      version_info = json.load(f)
  except FileNotFoundError:
    print >> sys.stderr, "version_info.json not found"
    sys.exit(1)

  pinned_versions = version_info["pinned-versions"]
  iree_version = pinned_versions["iree-compiler"]
  print(iree_version)

  command = f"python3 -m pip install . -f https://openxla.github.io/iree/pip-release-links.html --force-reinstall .[xla,cpu,test]"
  print(command)
  proc = subprocess.Popen(command.split(' '))
  _, err = proc.communicate()
  print(err)

if __name__ == "__main__":
    main()
