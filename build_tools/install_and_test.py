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
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description="Install and test iree-jax")
  args = parser.parse_args()

  project_root = os.path.dirname(os.path.dirname(__file__))

  # Load version_info.json.
  version_info_file = os.path.join(project_root, "version_info.json")
  try:
    with open(version_info_file, "rt") as f:
      version_info = json.load(f)
  except FileNotFoundError:
    version_info = {}

  # We test at head by removing pinned versions.
  if "pinned-versions" in version_info:
    print("*** Removing pinned versions (test from head) ***")
    del version_info["pinned-versions"]

  print("*** Installing iree-jax...")
  subprocess.check_call(
      [
          sys.executable,
          "-m",
          "pip",
          "install",
          "-f",
          "https://github.com/google/iree/releases",
          "--force-reinstall",
          ".[xla,cpu,test]",
          "filecheck",
      ],
      cwd=project_root)

  pinned_versions = get_pinned_versions()
  print("*** INSTALLED VERSIONS:", pinned_versions)

  print("*** Running tests...")
  subprocess.check_call(["lit", "tests/"], cwd=project_root)

  print("*** ALL TESTS PASS ***")
  version_info["pinned-versions"] = pinned_versions

  version_info_json = json.dumps(version_info, indent=2)
  print(f"*** WRITE {version_info_file} ***")
  print(version_info_json)

  with open(version_info_file, "wt") as f:
    f.write(version_info_json)


def get_pinned_versions():
  kept_versions = [
      "iree-compiler", "iree-runtime", "iree-tools-xla", "jax", "jaxlib"
  ]
  freeze_str = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
  freeze_lines = freeze_str.decode("utf-8").splitlines()
  versions = {}
  for line in freeze_lines:
    comps = line.split("==", 1)
    if len(comps) == 2:
      if comps[0] in kept_versions:
        versions[comps[0]] = comps[1]
  return versions


if __name__ == "__main__":
  main()
