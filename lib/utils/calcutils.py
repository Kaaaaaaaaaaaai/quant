from .._ticker import Ticker
from .._portfolio import Portfolio
import jax.numpy as jnp
import numpy as np
from typing import Iterable
import pandas as pd
import polars as pl
import cupy as cp
from cupy import dot as cdot
from cupy import eye as ceye
from numpy import dot, eye

def calc_stats(portfolio:Portfolio, device:str = "cpu") -> dict:

    returns = np.array([ticker.returns for ticker in portfolio.tickers])
    expected_return = portfolio.weights @ returns
    volatility = np.sqrt(portfolio.weights.T @ np.array(calc_sigma(portfolio, device=device)) @ portfolio.weights)
    sharpe_ratio = expected_return / volatility if volatility != 0 else 0.0

    return {
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio
    }

def calc_sigma(portfolio:Portfolio, device:str = "cpu") -> jnp.ndarray:

    s, rho = np.array([ticker.volatility for ticker in portfolio.tickers]), __corr_matrix(portfolio.tickers, device=device)
    return jnp.array(rho * (s[:, None] @ s[None, :]))

def __corr_matrix(data:Iterable[Ticker], device:str) -> np.ndarray:

    if device not in ["cpu", "gpu"]:
        raise ValueError("Device must be either 'cpu' or 'gpu'.")

    x = pd.concat([obj.history.select(["Date", "Price"]).rename({"Price":obj.id}).to_pandas().set_index("Date") for obj in data], axis=1, join="outer")
    x = pl.from_pandas(x, include_index=False).fill_null(strategy="forward").drop_nulls().to_numpy()
    x = x - x.mean(axis=0, keepdims=True)

    n = x.shape[1]

    if device == "gpu":
        if not cp.cuda.is_available():
            raise RuntimeError("CuPy is not configured to use CUDA. Check your CuPy installation.")
        x = cp.array(x)
        return cp.asnumpy(cp.divide(cdot(x.T, x), cp.sqrt(cdot(cdot((cdot(x.T, x) * ceye(n)), cp.ones((n, n))), (cdot(x.T, x) * ceye(n))))))

    else:
        return np.divide(dot(x.T, x), np.sqrt(dot(dot((dot(x.T, x) * eye(n)), np.ones((n, n))), (dot(x.T, x) * eye(n)))))