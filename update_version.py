
import git
import json
import os.path
import sys
import time

def main(args):
    dirpath = os.path.dirname(__file__)
    with open(os.path.join(dirpath, "version_info.json"), 'r') as f:
        js = json.load(f)

    repo = git.Repo(dirpath)
    commit = repo.head.commit.committed_date 

    package_version = time.strftime("%Y%m%d", time.gmtime(commit))

    if (len(args) > 0):
        package_version

    js["package-version"] = package_version
    js["package-suffix"] = ""

    with open(os.path.join(dirpath, "version_info.json"), 'w') as f:
        f.write(json.dumps(js, indent=2))
        f.write("\n")

if (__name__ == "__main__"):
    main(sys.argv[1:])
