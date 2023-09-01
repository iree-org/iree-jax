# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description='Determine setup options.')
parser.add_argument('--use-pinned', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--add-version')

'''Sets values in version_info.json

Sets values related to the libraries version numbers, the pinned versions, and
whether the pinned versions should be exact or a lower bound. This is used for
configuring on demand for bumping or validating pinned versions.
'''
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

  args = parser.parse_args()
  version_info["use-pinned"] = args.use_pinned

  if args.add_version:
    version_info["package-version"] = args.add_version
  elif "package-version" in version_info:
    del version_info["package-version"]

  print(json.dumps(version_info, indent=2))

  with open(os.path.join(project_root, "version_info.json"), 'w') as f:
      f.write(json.dumps(version_info, indent=2))
      f.write("\n")

if __name__ == "__main__":
  main()

