from iree.jax import Program
import jax.numpy as jnp
import jax

x = jnp.ones((), jnp.float32)

class GradProgram(Program):
  def run(self, x=Program.like(x)):
    return self.double_prim(x)

  @Program.kernel
  def double_prim(x):
    def double(x):
      return x * 2
    return jax.grad(double)(x)

m = GradProgram()
print(Program.get_mlir_module(m))
