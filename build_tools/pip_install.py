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

  command = f"python3 -m pip install . -f https://iree-org.github.io/iree/pip-release-links.html --force-reinstall .[xla,cpu,test]"
  print(command)
  proc = subprocess.Popen(command.split(' '))
  _, err = proc.communicate()
  print(err)


if __name__ == "__main__":
  main()
