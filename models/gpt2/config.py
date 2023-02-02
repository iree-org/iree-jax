'''Flags and model configuration details, useful for replicating across files.'''
from absl import flags
from os import path

import collections
import pathlib

assets_dir = path.join(pathlib.Path(__file__).parent.resolve(), "assets")

flags.DEFINE_string('binary_path', '/tmp/gpt2.vmfb', 'Path for binary')
flags.DEFINE_string('ir_path', '/tmp/gpt2.mlir', 'Path for IR')
flags.DEFINE_string('assets_path', assets_dir, 'Path for assets dir')


# Create a tuple with model configuration details as follows:
# B - batch size
# K - encoder sequence length
# S - total sequence length
# T - decode step size
def get_config():
  config = collections.namedtuple('Config', ['B', 'K', 'S', 'T'])
  return config(4, 8, 64, 1)
