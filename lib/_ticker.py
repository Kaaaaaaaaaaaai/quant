import polars as pl

class Ticker:

    def __init__(self, ticker: str):

        self.id = ticker
        self.returns = 0.0
        self.volatility = 0.0
        self.dividends = 0.0
        self.cost = 0.0
        self.history = pl.DataFrame()

    @classmethod
    def load(cls, ticker:str, data_dir:str, date_col:str, target_col:str, dividend_col:str = None, cost:float = 0.0, date_regx:str = r"%Y-%m-%d", *args) -> "Ticker":

        data = pl.read_csv(data_dir).rename(
            {date_col: "Date", target_col: "Price"}
            ).with_columns(
                pl.col("Date").str.strptime(pl.Date, date_regx),
                (pl.col("Price") / pl.col("Price").shift(1) - 1).alias("capital_return")
            )
        
        instance = cls(ticker)
        instance.id = ticker
        instance.cost = cost
        instance.history = data.select(["Date", "Price"])
        instance.returns = data.select(pl.col("capital_return").mean()).item() * 246
        instance.volatility = data.select(pl.col("capital_return").std(ddof = 1)).item() * 15.684387141358123   # np.sqrt(246)

        if dividend_col:
            years = data.select(pl.col("Date").max()).item().year - data.select(pl.col("Date").min()).item().year
            dividend_per_year = data.filter(pl.col(dividend_col) > 0).shape[0] / years if years > 0 else 0
            instance.dividends = data.select(
                (pl.col(dividend_col) / pl.col("Price").shift(-1)).drop_nulls()
            ).mean().item() * dividend_per_year

        return instance