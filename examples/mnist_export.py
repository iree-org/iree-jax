# Copyright 2021 Google LLC
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

# Usage: python high_level_mnist_export.py {dest dir}

import os
import sys

import numpy.random as npr
from examples import datasets

from iree.compiler import (
    tools as iree_tools,)

import jax.core
import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from jax.interpreters.xla import abstractify

from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           register_pytree_node)

from iree.jax import (
    like,
    kernel,
    IREE,
    Program,
)


def main(args):
  output_dir = args[0]
  os.makedirs(output_dir, exist_ok=True)
  module = build_model()

  print("Saving mlir...")
  with open(os.path.join(output_dir, "mnist_train.mlir"), "wb") as f:
    Program.get_mlir_module(module).operation.print(f, binary=True)

  print("Compiling binary...")
  binary = IREE.compile_program(module)

  print("Saving binary...")
  with open(os.path.join(output_dir, "mnist_train.vmfb"), "wb") as f:
    f.write(binary.compiled_artifact)


def build_model():
  init_random_params, predict = stax.serial(
      Dense(1024),
      Relu,
      Dense(1024),
      Relu,
      Dense(10),
      LogSoftmax,
  )

  def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))

  rng = random.PRNGKey(0)
  _, init_params = init_random_params(rng, (-1, 28 * 28))
  opt_init, opt_update, opt_get_params = optimizers.momentum(0.001, mass=0.9)
  opt_state = opt_init(init_params)

  example_batch = get_example_batch()

  class MnistModule(Program):
    # We don't want to export the host-side initial values, so export those
    # first and disable initialization.
    _params = init_params
    _opt_state = opt_state

    def get_params(self):
      return self._params

    def get_opt_state(self):
      return self._opt_state

    def set_opt_state(self, new_opt_state=like(opt_state)):
      self._opt_state = new_opt_state

    def initialize(self, rng=like(rng)):
      self._opt_state = self._initialize_optimizer(rng)

    def update(self, batch=like(example_batch)):
      new_opt_state = self._update_step(batch, self._opt_state)
      self._opt_state = new_opt_state

    def predict(self, inputs=like(example_batch[0])):
      return self._predict_target_class(self._params, inputs)

    @kernel
    def _initialize_optimizer(rng):
      _, init_params = init_random_params(rng, (-1, 28 * 28))
      return opt_init(init_params)

    @kernel
    def _update_step(batch, opt_state):
      params = opt_get_params(opt_state)
      return opt_update(0, grad(loss)(params, batch), opt_state)

    @kernel
    def _predict_target_class(params, inputs):
      predicted_class = jnp.argmax(predict(params, inputs), axis=1)
      return predicted_class

  return MnistModule()


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


main(sys.argv[1:])
