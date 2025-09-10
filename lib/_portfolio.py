from typing import Iterable
import polars as pl
import numpy as np
from ._ticker import Ticker

class Portfolio:

    def __init__(self, tickers:Iterable[Ticker], weights:np.ndarray) -> None:
      
        if len(tickers) != len(weights):
            raise ValueError("The number of tickers must match the number of weights.")

        self.tickers = tickers

        self._weights = np.clip(weights, a_min = 0.0, a_max = np.inf)
        self._weights = self._weights / np.sum(self._weights)
    
    def __repr__(self) -> str:
        return pl.DataFrame({"Ticker": [ticker.id for ticker in self.tickers],"Weight": self.weights}).__repr__()
    
    @property
    def weights(self) -> np.ndarray:
        return self._weights
    
    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        if len(value) != len(self.tickers):
            raise ValueError("The number of weights must match the number of tickers.")
        
        self._weights = np.clip(value, a_min = 0.0, a_max = np.inf)
        self._weights = self._weights / np.sum(self._weights)
    
    def to_csv(self, path:str) -> None:

        pl.DataFrame({
            "Ticker": [ticker.id for ticker in self.tickers],
            "Weight": self.weights
        }).sort(by="Weight", descending=True).filter(pl.col("Weight") != 0).write_csv(path)