from iree.jax import Program
import jax.numpy as jnp

x = jnp.ones((4, 4), jnp.float32)

class DoubleProgram(Program):
  def run(self, x=Program.like(x)):
    return self.mul(x)
    
  @Program.kernel
  def mul(x):
    return x * 2

m = DoubleProgram()
print(Program.get_mlir_module(m))
