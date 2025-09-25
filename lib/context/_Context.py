import jax.numpy as jnp
import jax
from functools import partial

class Context:

    def __init__(self):

        pass

    def minimize_target(self):

        pass

    def eq_constraints(self, x:jnp.ndarray):

        return jnp.array([])

    def ineq_constraints(self, x:jnp.ndarray):

        return jnp.array([])

    @partial(jax.jit, static_argnames=["self"])
    def constraints(self, x:jnp.ndarray) -> jnp.ndarray:

        return jnp.asarray((*self.ineq_constraints(x), *self.eq_constraints(x)))