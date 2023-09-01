# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Flags and model configuration details, useful for replicating across files."""
from absl import flags
from os import path

import collections
import pathlib

assets_dir = path.join(pathlib.Path(__file__).parent.resolve(), "assets")

flags.DEFINE_string("binary_path", "/tmp/gpt2.vmfb", "Path for binary")
flags.DEFINE_string("ir_path", "/tmp/gpt2.mlir", "Path for IR")
flags.DEFINE_string("assets_path", assets_dir, "Path for assets dir")
flags.DEFINE_boolean("no_compile", False, "Generate MLIR only, no IREE compling.")


flags.DEFINE_integer("batch_size", 4, "Minibatch size", lower_bound=1)
flags.DEFINE_integer(
    "encoder_sequence_length", 8, "Encoder sequence length", lower_bound=1
)
flags.DEFINE_integer(
    "total_sequence_length", 64, "Total sequence length", lower_bound=1
)
flags.DEFINE_integer("decode_step_size", 1, "Decode step size", lower_bound=1)


# Create a tuple with model configuration details as follows:
# B - batch size
# K - encoder sequence length
# S - total sequence length
# T - decode step size
def get_config():
    config = collections.namedtuple("Config", ["B", "K", "S", "T"])
    FLAGS = flags.FLAGS
    return config(
        FLAGS.batch_size,
        FLAGS.encoder_sequence_length,
        FLAGS.total_sequence_length,
        FLAGS.decode_step_size,
    )
