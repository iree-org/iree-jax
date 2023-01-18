'''Tests the exported gpt-2 model against a set of known decode steps.'''

from absl import flags
from absl.testing import absltest
from os import path
from transformers import GPT2Tokenizer

import builtins
import iree.compiler.tools as compiler
import iree.runtime as iree_rt
import numpy as np
import pathlib

import config

FLAGS = flags.FLAGS


class ExportedModelTest(absltest.TestCase):

  def setUp(self):
    gpt2_dir = FLAGS.assets_path
    self.tokenizer = GPT2Tokenizer(vocab_file=path.join(gpt2_dir, 'vocab.json'),
                                   merges_file=path.join(
                                       gpt2_dir, 'merges.txt'))
    self.tokenize = self.tokenizer.encode

    with open(FLAGS.binary_path, 'rb') as f:
      config = iree_rt.Config("local-task")
      context = iree_rt.SystemContext(config=config)
      vm_module = iree_rt.VmModule.from_flatbuffer(config.vm_instance, f.read())
      context.add_vm_module(vm_module)
      self.module = context.modules.gpt2_module
      self.encode = self.module.encode
      self.decode = self.module.decode

  def makeInput(self, prompts):
    if not isinstance(prompts, list):
      prompts = [prompts]
    cfg = config.get_config()
    B = cfg.B
    K = cfg.K
    ids = np.zeros((B, K), dtype=np.int32)
    length = np.zeros((B,), dtype=np.int32)
    for i, prompt in enumerate(prompts):
      tokenized = self.tokenize(prompt)
      ids[i, :len(tokenized)] = tokenized
      length[i] = len(tokenized)
    return ids, length

  def test_counting(self):
    ids, lengths = self.makeInput('zero one two three four')

    x0 = np.asarray(self.encode(ids, lengths))[0, 0]
    x1 = np.asarray(self.decode())[0, 0]
    x2 = np.asarray(self.decode())[0, 0]

    e0 = self.tokenize(' five')[0]
    e1 = self.tokenize(' six')[0]
    e2 = self.tokenize(' seven')[0]

    self.assertEqual(x0, e0)
    self.assertEqual(x1, e1)
    self.assertEqual(x2, e2)

  def test_words(self):
    ids, lengths = self.makeInput('when in the course')

    x0 = np.asarray(self.encode(ids, lengths))[0, 0]
    x1 = np.asarray(self.decode())[0, 0]
    x2 = np.asarray(self.decode())[0, 0]

    e = self.tokenize(' of a long')
    e0 = e[0]
    e1 = e[1]
    e2 = e[2]

    self.assertEqual(x0, e0)
    self.assertEqual(x1, e1)
    self.assertEqual(x2, e2)

  def test_batch(self):
    inputs = ['zero one two three four', 'when in the course']
    ids, lengths = self.makeInput(inputs)

    x0 = np.asarray(self.encode(ids, lengths))[:, 0]
    x1 = np.asarray(self.decode())[:, 0]
    x2 = np.asarray(self.decode())[:, 0]

    y0 = [x0[0], x1[0], x2[0]]
    y1 = [x0[1], x1[1], x2[1]]

    e0 = self.tokenize(' five six seven')
    e1 = self.tokenize(' of a long')

    self.assertEqual(y0, e0)
    self.assertEqual(y1, e1)


if __name__ == '__main__':
  absltest.main()
