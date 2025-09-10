from ._Context import Context
from .._portfolio import Portfolio
import jax.numpy as jnp

class SharpeContext(Context):

    def __init__(self, portfolio: Portfolio, risk_free_rate: float):

        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.portfolio = portfolio

    def minimize_target(self, x:jnp.ndarray) -> jnp.ndarray:

        return - (x.T @ self.returns - self.risk_free_rate) / jnp.sqrt(x.T @ (self.sigma) @ x)