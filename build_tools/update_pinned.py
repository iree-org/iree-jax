# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import subprocess
import sys
from subprocess import check_output

'''Update the version_info.json with the currently installed libraries.'''
def main():
  reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
  reqs = [c.split("==") for c in reqs.decode('ascii').split('\n') if "==" in c]
  reqs = {a : b for a, b in reqs }

  # Load version_info.json.
  project_root = os.path.dirname(os.path.dirname(__file__))
  version_info_file = os.path.join(project_root, "version_info.json")
  try:
    with open(version_info_file, "rt") as f:
      version_info = json.load(f)
  except FileNotFoundError:
    print >> sys.stderr, "version_info.json not found"
    sys.exit(1)

  pinned = version_info["pinned-versions"]
  for pinned_lib in pinned:
      if pinned_lib in reqs:
        pinned[pinned_lib] = reqs[pinned_lib]

  version_info = {a : version_info[a] for a in version_info if a == "pinned-versions"}

  with open(os.path.join(project_root, "version_info.json"), 'w') as f:
      f.write(json.dumps(version_info, indent=2))
      f.write("\n")
  print(pinned["iree-compiler"])

if  __name__ == "__main__":
  main()
