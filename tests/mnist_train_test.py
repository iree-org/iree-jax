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

import jax
import os
import numpy.random as npr
from examples import datasets
from iree import runtime as iree_rt
import jax.core
import jax.numpy as jnp
from jax import grad, random
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from jax.tree_util import tree_flatten
from iree.jax import (
    like,
    kernel,
    IREE,
    Program,
)
from tempfile import TemporaryDirectory
import numpy as np
from typing import Any, Callable
import unittest


def get_example_batch():
  batch_size = 128
  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  batches = data_stream()
  return next(batches)


def get_model():
  init_random_params, predict = stax.serial(
      Dense(128),
      Relu,
      Dense(128),
      Relu,
      Dense(10),
      LogSoftmax,
  )
  return init_random_params, predict


def loss(params, batch, predict_fn):
  inputs, targets = batch
  preds = predict_fn(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))


def create_iree_jax_module():
  init_random_params, forward = get_model()

  rng = random.PRNGKey(12345)
  _, init_params = init_random_params(rng, (-1, 28 * 28))
  opt_init, opt_update, opt_get_params = optimizers.momentum(0.001, mass=0.9)
  opt_state = opt_init(init_params)

  example_batch = get_example_batch()

  class IreeJaxMnistModule(Program):
    _opt_state = opt_state

    def get_params(self):
      return opt_get_params(self._opt_state)

    def get_opt_state(self):
      return self._opt_state

    def set_opt_state(self, new_opt_state=like(opt_state)):
      self._opt_state = new_opt_state

    def initialize(self, rng=like(rng)):
      self._opt_state = self._initialize_optimizer(rng)

    def update(self, batch=like(example_batch)):
      new_opt_state = self._update_step(batch, self._opt_state)
      self._opt_state = new_opt_state

    def forward(self, inputs=like(example_batch[0])):
      return self._forward(opt_get_params(self._opt_state), inputs)

    @kernel
    def _initialize_optimizer(rng):
      _, init_params = init_random_params(rng, (-1, 28 * 28))
      return opt_init(init_params)

    @kernel
    def _update_step(batch, opt_state):
      params = opt_get_params(opt_state)
      return opt_update(0, grad(loss)(params, batch, forward), opt_state)

    @kernel
    def _forward(params, inputs):
      return forward(params, inputs)

  return IreeJaxMnistModule()


def build_iree_module(artifacts_dir,
                      backend: str = "llvm-cpu",
                      runtime: str = "local-task"):
  module = create_iree_jax_module()
  with open(os.path.join(artifacts_dir, "mnist_train.mlir"), "wb") as f:
    Program.get_mlir_module(module).operation.print(f, binary=True)
  binary = IREE.compile_program(module, backends=[backend], runtime=runtime)
  iree_vmfb_path = os.path.join(artifacts_dir, "mnist_train.vmfb")
  with open(iree_vmfb_path, "wb") as f:
    f.write(binary.compiled_artifact)
  loaded_module = iree_rt.system_api.load_vm_flatbuffer_file(iree_vmfb_path,
                                                             driver=runtime)
  return loaded_module


def build_jax_module():
  init_random_params, forward = get_model()

  rng = random.PRNGKey(12345)
  _, init_params = init_random_params(rng, (-1, 28 * 28))
  opt_init, opt_update, opt_get_params = optimizers.momentum(0.001, mass=0.9)
  opt_state = opt_init(init_params)

  example_batch = get_example_batch()

  class JaxMnistModule:
    _opt_state = opt_state

    def get_params(self):
      return opt_get_params(self._opt_state)

    def get_opt_state(self):
      return self._opt_state

    def set_opt_state(self, new_opt_state):
      self._opt_state = new_opt_state

    def initialize(self, rng):
      self._opt_state = JaxMnistModule._initialize_optimizer(rng)

    def update(self, batch):
      new_opt_state = JaxMnistModule._update_step(batch, self._opt_state)
      self._opt_state = new_opt_state

    def forward(self, inputs):
      return JaxMnistModule._forward(opt_get_params(self._opt_state), inputs)

    @jax.jit
    def _initialize_optimizer(rng):
      _, init_params = init_random_params(rng, (-1, 28 * 28))
      return opt_init(init_params)

    @jax.jit
    def _update_step(batch, opt_state):
      params = opt_get_params(opt_state)
      return opt_update(0, grad(loss)(params, batch, forward), opt_state)

    @jax.jit
    def _forward(params, inputs):
      return forward(params, inputs)

  return JaxMnistModule()


def assert_array_almost_equal(a, b):
  np_a = np.asarray(a)
  np_b = np.asarray(b)
  # Test for absolute error.
  np.testing.assert_array_almost_equal(np_a, np_b, decimal=5)
  # Test for relative error while ignoring false positives from
  # catastrophic cancellation.
  np.testing.assert_array_almost_equal_nulp(np.abs(np_a - np_b) + 10**-7,
                                            np.zeros_like(np_a),
                                            nulp=10**8)


def assert_array_list_equal(
    a,
    b,
    array_compare_fn: Callable[[Any, Any],
                               None] = np.testing.assert_array_equal):
  assert (len(a) == len(b))
  for x, y in zip(a, b):
    array_compare_fn(x, y)


def assert_array_list_almost_equal(a, b):
  assert_array_list_equal(a, b, assert_array_almost_equal)


def train_mnist_test(backend: str, runtime: str):
  """Run a training step on the same model with both Jax and IREE and
  verify that results are the same."""

  example_batch = get_example_batch()

  with TemporaryDirectory() as tmp_dir:
    iree_module = build_iree_module(artifacts_dir=tmp_dir,
                                    backend=backend,
                                    runtime=runtime)
    jax_module = build_jax_module()

    # Check state is the same
    assert_array_list_equal(iree_module.get_opt_state(),
                            tree_flatten(jax_module.get_opt_state())[0])

    # Check one training step.
    iree_module.update(*example_batch)
    jax_module.update(example_batch)
    assert_array_list_almost_equal(iree_module.get_opt_state(),
                                   tree_flatten(jax_module.get_opt_state())[0])

    # Check inference.
    iree_module.set_opt_state(*tree_flatten(jax_module.get_opt_state())[0])
    prediction_iree = iree_module.forward(example_batch[0])
    prediction_jax = jax_module.forward(example_batch[0])
    assert_array_almost_equal(prediction_iree, prediction_jax)

    # Check intialization.
    rng = random.PRNGKey(6789)
    iree_module.initialize(np.asarray(rng, dtype=np.int32))
    jax_module.initialize(rng)
    assert_array_list_almost_equal(iree_module.get_opt_state(),
                                   tree_flatten(jax_module.get_opt_state())[0])


class MnistTrainTest(unittest.TestCase):

  def test_train_mnist_cuda(self):
    train_mnist_test(backend="cuda", runtime="cuda")

  def test_train_mnist_llvm_cpu(self):
    train_mnist_test(backend="llvm-cpu", runtime="local-task")


if __name__ == "__main__":
  unittest.main()
