from lib.context import Context
from .Optimizer import Optimizer
import jax.numpy as jnp
import jax
import numpy as np

class BarrierOptimizer(Optimizer):

    def __init__(self, context:Context, mu:float=1.0, mu_shrink:float=0.1, step:float = 0.01, tol:float=1e-4, max_inner_iter:int=100, max_outer_iter:int=100, device:str = "cpu", enable_x64:bool = False) -> None:
        
        super().__init__()
        self.context = context
        self.mu = mu
        self.tol = tol
        self.inner_iter = max_inner_iter
        self.outer_iter = max_outer_iter
        self.step = step
        self.mu_shrink = mu_shrink
        self.device = device
        self.enable_x64 = enable_x64

    def optimize(self, initial_guess:jnp.ndarray|np.ndarray, eps:float) -> jnp.ndarray:
        
        jax.config.update("jax_platform_name", self.device)
        jax.config.update("jax_enable_x64", self.enable_x64)

        x = initial_guess
        mu = self.mu
        grad_norm = np.inf

        for _ in range(self.outer_iter):

            print(f"Outer iteration {_}, mu = {mu}")

            target_function = jax.jit(lambda x: self.context.minimize_target(x) - mu * jnp.sum(jnp.log(-self.context.constraints(x) + eps)))
            dfdx = jax.grad(target_function)

            grad_norm_old_outer = grad_norm

            for _ in range(self.inner_iter):

                grad_norm_old = grad_norm

                grad = dfdx(x).block_until_ready()

                x = x - self.step * grad

                if (grad_norm := float(jnp.linalg.norm(grad))) < self.tol:

                    print(f"Converged with ||grad|| = {grad_norm}")
                    break

                elif abs(1 - grad_norm / grad_norm_old) < self.tol:

                    print(f"Converged with ||grad|| change = {abs(1 - grad_norm / grad_norm_old)}")
                    break

                if _ % 10 == 0:
                    print(f"Inner iteration {_}, ||grad|| = {grad_norm}")

            if abs(1 - grad_norm / grad_norm_old_outer) < self.tol:
                print(f"Outer loop converged with ||grad|| change = {abs(1 - grad_norm / grad_norm_old_outer)}")
                break
            mu *= self.mu_shrink

        return x