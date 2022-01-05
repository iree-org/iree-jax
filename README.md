# IREE JAX API

*NOTE: This repository is currently under construction. Please stand by for this
message to be removed before considering it usable.*

The IREE JAX API provides a compiler and runtime bridge between
[JAX](https://github.com/google/jax) and [IREE](https://github.com/google/iree)
for the purpose of allowing programs to be extracted and compiled from JAX
for deployment with IREE, without a Python or JAX dependency.

In order to understand how this works, it is important to have a working
knowledge of what JAX and IREE programs are (which we discuss after the
example).

## By example

In order to introduce the concepts, we present a sample program which directly
compiles to IREE. It doesn't do anything interesting but is a place to
write comments and explanations.

```python
# Everything that is needed for basic use can be accessed from the Module
# class. Some additional imports may be needed for more advanced cases.
from iree.jax import Module

# Host-side arrays and trees of arrays can be mirrored into an IREE Module.
# All structure and any aliasing between trees is retained in the module.
x = jnp.ones((3, 4), jnp.float32) * 4.0
b = jnp.ones((3, 4), jnp.float32)

Params = namedtuple("Params", "x,b")

params = Params(x, b)


class TrivialKernel(Module):
  """By sub-classing `jax.iree.Module`, you create an IREE program which can
  be compiled (and executed in the host process) by instantiating it.

  The module's name is by default the snake_case equivalent of the class name
  with any "Module" suffix removed. It can be specified explicitly via:

    class MyModule(Module, export_name="foobar"):
  """

  # Globals are created by giving names in the class to arrays and trees of
  # arrays from the host Python session. When accessed on an instance, they
  # retain their structure but access values in the exported program.
  _params = params

  # The above is the default, sugared form of the more general mechanism
  # for exporting a global. Using this form allows you to create uninitialized
  # globals or annotate them as immutable (the compiler will generally figure
  # this out for you).
  # _params = Module.export_global(params, initialize=True, mutable=True)

  # Sometimes it is useful to give a name to an aliased part of a tree. This
  # is fully allowed. The first statement in the class which exports any
  # particular leaf will define its characteristics
  # (initialization/mutability/name). Subsequent exports will create aliases,
  # just like in the original Python session.
  _x = params.x

  # Any function defined without annotation becomes a public function in the
  # IREE program with an input signature derived from the declaration and
  # outputs derived by tracing. We call these "Module Procedures".
  # This function just provides an accessor to get the tree of _params.
  def get_params(self):
    return self._params

  # Here we see how to specify an input signature. The `Module.like` helper
  # is just producing a tree of `jax.core.AbstractValue` representing some
  # reference from the host program. This is an easy way to represent such
  # details, but users are also free to constract AbstractValue trees
  # themselves.
  def run(self, multiplier=Module.like(x)):
    # When tracing a public function, the primary thing you can do is call
    # immutable kernel functions using combinations of function arguments and
    # module globals.
    result = self._linear(multiplier, self._params.x, self._params.b)
    # Module globals can be updated directly via assignment. This is a sugared
    # form of the more general `Module.store_global()` helper.
    self._x = result
    return result

  # Public functions can also accept arbitrary trees.
  def set_params(self, new_params=Module.like(params)):
    # And trees of globals can be updated in one assignment.
    self._params = new_params

  # "Kernel" functions are basically equivalent to `jax.jit` but specially
  # wrapped so that they can exist as members of a Module. They act like
  # staticmethods (i.e. they do not take a `self`) and they can only operate on
  # arguments or host process state that is legal for `jax.jit`. Think of them
  # as private static functions of the module, and by convention we name them
  # with a leading underscore.
  @Module.kernel
  def _linear(m, x, b):
    return m * x + b

# Instantiating a module will trace it and invoke the `ireec` compiler on it.
# keyword arguments control compilation behavior. The defaults should be
# reasonable for running directly on the host.
m = TrivialKernel()

# You can inspect the MLIR module which was extracted.
print(Module.get_mlir_module(m))

# While the primary purpose of extracting a program is to run it *elsewhere*,
# you can't spell "ML" without "debugging", and instantiated modules are fully
# usable interactively. Under the hood, the compiled artifact is loaded into
# the IREE runtime and wrappers are created to dispatch to named public
# functions.
print("Initial params:", m.get_params())

# Stateful updates can be made.
update = jnp.ones_like(x)
print("Run:", m.run(update))

# And inspected.
print("Updated params:", m.get_params())

# You can save off the compiled artifact to run elsewhere.
Module.get_compiled_artifact(m).save("/tmp/some_file.vmfb")
```

## What is a JAX program?

The challenge with talking about extracting a JAX program is that in all
generality, a JAX program is bounded only by what can run in the host Python
interpreter. What we are looking for is a simpler definition that allows a
useful set of standalone programs to be constructed and that meshes well with
the programming model employed by typical JAX developers.

The components that are the most interesting towards this end are:

* `jax.jit` functions: stateless functions mapping inputs to outputs, natively
  represented as JAXPR IR and universally convertible to MHLO IR.
* Multi-dimensional arrays very similar to numpy arrays, with extensions to
  enable tracing and distribution.
* Trees of arrays and primitives (a structured generalization of nested dicts,
  lists and tuples).
* "AbstractValues" mirroring all concrete data types, facilitating symbolic
  tracing and descriptions of program elements.

There are of course many more details than this, but these are the components
we will assemble.

## What is an IREE program?

An IREE program is:

* Defined by an MLIR module containing:
  * A name, which can be used for cross-referencing between modules. All modules
    loaded into a single runtime instance must have a unique name.
  * Zero or more globals. Globals can contain any scalar or multi-dimensional
    array of any supported type (IREE has an open type system and supports
    more than this but these types are relevant to JAX). Globals can be:
      * Loaded, yielding their current value
      * Stored, updating the value
      * Mutable or immutable
      * Initialized, either via a constant or via special initializer functions
        that can do whatever they want at module load time
  * One or more public functions, each of which accepts and returns scalars and
    buffers representationing multi-dimensional arrays. IREE functions can
    do a lot of things, but relevant to JAX, they can include:
      * A graph of MHLO operations (the native IR of JAX jitted functions)
      * Operations and types from
        [IREE's input dialect](https://github.com/google/iree/blob/main/llvm-external-projects/iree-dialects/include/iree-dialects/Dialect/Input/InputOps.td)
      * Operations from low level MLIR dialects (for programs coming from a high
        level IR like MHLO, these are typically not used, but they provide
        the generality needed for more advanced cases such as implementing
        full language compilers):
        * MLIR [arith](https://mlir.llvm.org/docs/Dialects/ArithmeticOps/) dialect
        * MLIR [linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) dialect
        * MLIR [math](https://mlir.llvm.org/docs/Dialects/MathOps/) dialect
        * MLIR [std](https://mlir.llvm.org/docs/Dialects/Standard/) dialect (being phased out)
        * MLIR [tensor](https://mlir.llvm.org/docs/Dialects/TensorOps/) dialect
* Compiled by the `ireec` compiler, targeting a number or architectures and
  devices
* Compiled into an "IREE VM Program" which can be:
  * Serialized to a `vmfb` (VM Flatbuffer) for direct execution by the IREE
    runtime.
  * Serialized to a C program for use in contexts where a runtime implementation
    is not desirable.
* Intended to run on a "single node" (single SOC, single set of devices on the
  same bus, etc). Horizontal distribution is intended to be a layer atop IREE
  if desired.

## Module Procedures

As mentioned above, "Module Procedures" are the public functions that can be
invoked on a compiled Module. Today, they are implemented with a limited
tracer which defines them by run. A Python based compiler is under development
to allow more sophisticated procedures to be represented.

The following are allowed in a traced procedure (this is restricted in order
to maintain a small surface area for a future compiler-based approach -- in
reality, there are many ways to hack the current system to do more):

* Referencing Module globals and kernel functions by resolving attributes
  against `self`.
* Accessing nested dict/list/tuple/namedtuple members of the above with
  attribute notation (i.e. `parent.child`) or indexing (i.e. `parent[0]`).
* Calling a kernel function with arguments derived from the above.
* Assigning to a global.
* Returning values.
* Using one of the builtin intrinsics:
  * `Module.store_global`
  * `Module.print` (TODO)


# Development

These are WIP instructions for getting a functioning development setup. We
aim to improve this and make it more turnkey over time.

Pip installable releases are not yet available. However, this project is
pure Python and can be installed locally for development (note that this
pulls from IREE pre-release snapshots):

```shell
python -m pip install -e . -f https://github.com/google/iree/releases
```

Note that in order to function the version of MLIR and MHLO used in the
installed jaxlib must be syntax compatible with the versions used in IREE.
For releases, we synchronize these, but for development it can drift and
cause errors.

The easiest way to ensure this is to pin the JAX tensorflow version to the
version that IREE was compiled with and follow the JAX instructions to build
jaxlib locally. See:

* [Building jaxlib from source](https://jax.readthedocs.io/en/latest/developer.html#building-jaxlib-from-source)
* [Instructions for updating TensorFlow in JAX](https://github.com/google/jax/blob/main/WORKSPACE#L8)
  (this is often optional as someone has likely committed an update).
* [IREE's pinned commits](https://github.com/google/iree/blob/main/SUBMODULE_VERSIONS.txt)

You may need to manually `pip uninstall` the automatically installed jaxlib.

## Running Tests

`lit tests/`

## Using an IREE development tree

Sometimes you will want to build with a local IREE. From IREE's build
directory:

```
source .env && export PYTHONPATH
```

You must have built IREE with the `-DIREE_BUILD_LEGACY_JAX=OFF` to disable
the original bundled JAX API.

You may need to pip uninstall the automatically installed
`iree-compiler-snapshot` and `iree-runtime-snapshot` packages.

For IDE integration, you may just want to copy IREE's `.env` file to the
root of this repo if working in this mode.
