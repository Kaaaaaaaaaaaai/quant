import polars as pl

"""
Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
xxxx-xx-xx, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 1
"""

def clean_stock_data(dir:str) -> None:

    df = pl.read_csv(dir)

    if "Date" in df.columns:

        df = df.with_columns(pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d 00:00:00+09:00").alias("Date"))

    l = 0
    for row in df.iter_rows():

        if row[5] == 0:

            l += 1

        else:

            break
    
    if l > 0:

        df = df.slice(l, df.height - l)

    df = df.fill_null(strategy = "forward")
    
    df.write_csv(dir)

    return

def clean_trust_data(dir:str) -> None:

    df = pl.read_csv(dir)

    if "年月日" in df.columns:

        df = df.with_columns(pl.col("年月日").str.strptime(pl.Date, "%Y年%m月%d日").alias("年月日"))

    df = df.rename({"基準価額(円)": "基準価額", "純資産総額（百万円）": "純資産総額"})

    df = df.fill_null(strategy = "forward")

    df.write_csv(dir)
    
    return

if __name__ == "__main__":

    import glob
    from concurrent.futures import ProcessPoolExecutor

    stock_files = glob.glob("data/stock/*.csv")
    trust_files = glob.glob("data/trust/*.csv")

    assert len(stock_files) > 0, "No stock data files found."
    assert len(trust_files) > 0, "No trust data files found."

    with ProcessPoolExecutor(max_workers = 18) as executor:

        print("Cleaning stock data...")
        executor.map(clean_stock_data, stock_files)
        print("Cleaning trust data...")
        executor.map(clean_trust_data, trust_files) 