import numpy as np
import polars as pl
import glob
from lib import Ticker, Portfolio
from lib.context import RiskContext
from lib.optimize import BarrierOptimizer
from lib.utils.calcutils import calc_sigma, calc_stats
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

def main():

    parser = ArgumentParser(description="Portfolio Optimization Script")
    parser.add_argument("--input_path", "-i", type=str, required = True, help="Path to input data folder")
    parser.add_argument("--output_path", "-o", type=str, required = True, help="Path to save optimized portfolio weights and logs")
    parser.add_argument("--num_workers", "-n", type=int, default=1, help="Number of workers for parallel processing")
    parser.add_argument("--risk_free_rate", "-r", type=float, default=0.0, help="Risk-free rate for Sharpe ratio calculation")
    parser.add_argument("--target_return", "-t", type=float, default=None, help="Target return for portfolio optimization")
    parser.add_argument("--outlier_threshold", "-ot", type=float, default=2.0, help="Threshold for outlier removal")
    parser.add_argument("--mu", "-m", type=float, default=1.0, help="Initial barrier parameter")
    parser.add_argument("--mu_shrink", type=float, default=0.1, help="Barrier parameter shrinkage factor")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for optimization convergence")
    parser.add_argument("--eps", type=float, default=1e-4, help="Small constant to avoid log(0) in barrier method")
    parser.add_argument("--max_inner_iter", type=int, default=1000, help="Maximum number of inner iterations for optimization")
    parser.add_argument("--max_outer_iter", type=int, default=100, help="Maximum number of outer iterations for optimization")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for optimization")
    parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "gpu"], help="Device to use for corrlation matrix calculations (cpu or gpu)")
    parser.add_argument("--enable_x64", "-x", action="store_true", help="Enable 64-bit precision in JAX computations")
    args = parser.parse_args()

    print("Loading data...")

    inputs = []
    df = pl.read_csv("resources/all_trust.csv").filter(pl.col("インデックス対象") == "〇").select(["協会コード", "信託報酬"])

    for file in glob.glob(f"{args.input_path}/*.csv"):
        ticker_id = file.split("/")[-1][:-4]
        if ticker_id in df["協会コード"].to_list():
            inputs.append((ticker_id, file, "年月日", "基準価額", None, df.filter(pl.col("協会コード") == ticker_id).select("信託報酬").item(), r"%Y-%m-%d"))
        

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:

        futures = [executor.submit(Ticker.load, *ip) for ip in inputs]

    tickers = sorted([future.result() for future in as_completed(futures)], key =lambda x: x.id)
    print(f"Loaded {len(tickers)} tickers.")

    vols = np.array([ticker.volatility for ticker in tickers])
    iq = np.quantile(vols, [0.75, 0.25])
    iqr = iq[0] - iq[1]
    upper_bound = iq[0] + args.outlier_threshold * iqr

    for ticker in tickers:
        if ticker.volatility > upper_bound:
            tickers.remove(ticker)

    print(f"Filtered to {len(tickers)} tickers.")

    portfolio = Portfolio(tickers, np.random.normal(0, 1, len(tickers)))
    r = np.array([ticker.returns for ticker in tickers])
    sigma = calc_sigma(portfolio, device=args.device)

    context = RiskContext(portfolio, args.target_return, sigma, r, args.risk_free_rate)
    optimizer = BarrierOptimizer(context=context, mu=args.mu, mu_shrink=args.mu_shrink, step=args.lr, tol=args.tol, max_inner_iter=args.max_inner_iter, max_outer_iter=args.max_outer_iter, device=args.device, enable_x64=args.enable_x64)

    print("Optimizing portfolio...")
    optimal_weights = optimizer.optimize(initial_guess = portfolio.weights, eps=args.eps)
    portfolio.weights = optimal_weights

    stats = calc_stats(portfolio, device=args.device)
    print(f"expected return: {stats['expected_return']:.4f}, Volatility: {stats['volatility']:.4f}, Sharpe Ratio: {stats['sharpe_ratio']:.4f}")

    portfolio.to_csv(f"{args.output_path}/optimized_portfolio.csv")

if __name__ == "__main__":
    
    main()