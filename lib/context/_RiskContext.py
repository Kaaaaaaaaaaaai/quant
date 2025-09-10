from ._Context import Context
from .._portfolio import Portfolio
import jax.numpy as jnp

class RiskContext(Context):

    def __init__(self, portfolio: Portfolio, target_return: float, sigma: jnp.ndarray, r: jnp.ndarray, risk_free_rate:float) -> None:

        super().__init__()
        self.target_return = target_return
        self.portfolio = portfolio
        self.sigma = sigma
        self.r = r
        self.risk_free_rate = risk_free_rate

    def minimize_target(self, x:jnp.ndarray) -> jnp.ndarray:

        return jnp.sqrt(x.T @ self.sigma @ x)

    def eq_constraints(self, x:jnp.ndarray) -> jnp.ndarray:

        return jnp.array([sum(x) - 1.0])

    def ineq_constraints(self, x:jnp.ndarray) -> jnp.ndarray:

        return jnp.array([self.target_return+self.risk_free_rate-(x.T @ self.r), *(0.0-x)])