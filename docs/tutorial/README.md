# IREE-JAX Tutorial

This tutorial shows you how to write JAX programs and turn them into MLIR so that the IREE compiler can turn them into vmfb for the IREE runtime to run.

## Quick Start

When you run each of these example programs, the MLIR is printed. If you sent the output to `iree-compiler`, the vmfb file would be made. For instance, the command below generates `/tmp/a.vmfb` from `01-double.py`.

```shell
python 01-double.py | \
  ./build/compiler/install/bin/iree-compile \
  --iree-input-type=mhlo \
  --iree-hal-target-backends=vmvx \
  - > /tmp/a.vmfb
```
