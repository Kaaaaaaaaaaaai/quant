import numpy as np
import polars as pl
from lib.downloads import download_data
from data_cleaning import clean_trust_data
from lib import Ticker, Portfolio
from lib.context import SharpeContext
from lib.optimize import BarrierOptimizer
from lib.utils.calcutils import calc_sigma, calc_stats
import requests

def trust_download(associ_fund_code: str, download_dir:str) -> None:

    url = f"https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000GKC6&associFundCd={associ_fund_code}"
    
    try:

        response = requests.get(url)

        if response.status_code == 200:

            content = response.text

            with open(f"{download_dir}/{associ_fund_code}.csv", "w", encoding = "utf-8") as f:

                f.write(content)

        else:

            print(f"Failed to download trust data: {response.status_code}")

    except Exception as e:

        print(f"Error downloading {associ_fund_code}: {e}")

    return

def main():

    risk_free_rate = 0.017
    mu = 1.0
    mu_shrink = 0.1
    tol = 1e-4
    max_inner_iter = 100
    max_outer_iter = 100
    lr = 0.1
    eps = 1e-6
    device = "gpu"
    enable_x64 = False

    targets = ["64316223", "55316241", "03317172", "03316183", "0331119A", "0331219A", "8931A236"]
    # for ticker in targets:trust_download(ticker, "data/trust")
    # for ticker in targets:clean_trust_data(f"data/trust/{ticker}.csv")
    df = pl.read_csv("resources/all_trust.csv").filter(pl.col("協会コード").is_in(targets)).select(["協会コード", "信託報酬"])

    if len(df) != len(targets):

        print(df.select("協会コード").to_series().to_list())
        raise ValueError("Some target tickers are not found in the dataset.")

    portfolio = Portfolio([Ticker.load(ticker, f"data/trust/{ticker}.csv", "年月日", "基準価額", None, df.filter(pl.col("協会コード") == ticker).select("信託報酬").item()/100, r"%Y-%m-%d") for ticker in targets], np.random.normal(0, 1, len(targets)))
    r = np.array([ticker.return_ for ticker in portfolio.tickers])
    sigma = calc_sigma(portfolio, device=device)

    context = SharpeContext(portfolio, sigma, r, risk_free_rate)
    optimizer = BarrierOptimizer(context=context, mu=mu, mu_shrink=mu_shrink, step=lr, tol=tol, max_inner_iter=max_inner_iter, max_outer_iter=max_outer_iter, device=device, enable_x64=enable_x64)

    print("Optimizing portfolio...")
    optimal_weights = optimizer.optimize(initial_guess = portfolio.weights, eps = eps)
    portfolio.weights = optimal_weights

    stats = calc_stats(portfolio, device=device)
    print(f"expected return: {stats['expected_return']:.4f}, Volatility: {stats['volatility']:.4f}, Sharpe Ratio: {stats['sharpe_ratio']:.4f}")

    portfolio.to_csv(f"results/portfolio_results/optimized_portfolio.csv")

if __name__ == "__main__":
    
    main()