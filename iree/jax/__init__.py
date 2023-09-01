# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .iree import IREE
from .program_api import Program

# Re-export some Module static methods so people can use them unqualified if
# desired.
like = Program.like
kernel = Program.kernel
store_global = Program.store_global
