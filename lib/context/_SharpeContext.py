from ._Context import Context
from .._portfolio import Portfolio
import jax.numpy as jnp

class SharpeContext(Context):

    def __init__(self, portfolio: Portfolio, sigma: jnp.ndarray, r: jnp.ndarray, risk_free_rate:float):

        super().__init__()
        self.portfolio = portfolio
        self.sigma = sigma
        self.r = r
        self.risk_free_rate = risk_free_rate

    def eq_constraints(self, x:jnp.ndarray) -> jnp.ndarray:

        return jnp.array([sum(x) - 1.0])

    def ineq_constraints(self, x:jnp.ndarray) -> jnp.ndarray:

        return jnp.array([self.risk_free_rate-(x.T @ self.r), *(0.0-x)])

    def minimize_target(self, x:jnp.ndarray) -> jnp.ndarray:

        return - (x.T @ self.r - self.risk_free_rate) / jnp.sqrt(x.T @ (self.sigma) @ x)