import glob
import os
import polars as pl
import datetime
from concurrent.futures import ProcessPoolExecutor

def filter_stock_data(file:str) -> None:

    df = pl.read_csv(file)
    symbol = file.split("/")[-1].split(".")[0]

    # check if the dataset contains sufficient entries
    if df.shape[0] < 120:

        print(f">{symbol}, insufficient entries")
        os.remove(file)
    
    # check if the stock has sufficient volume
    elif df["Volume"].mean() < 5000:

        print(f">{symbol}, insufficient volume")
        os.remove(file)

    # check if the stock is traded in regular basis (1)
    elif (df.filter(pl.col("Volume") == 0).shape[0] / df.shape[0]) > 0.15:

        print(f">{symbol}, too many zero volume")
        os.remove(file)

    # check if the stock has traded in regular basis (2)
    # elif (df["Volume"].std() / df["Volume"].mean()) > 10:

    #     print(f">{symbol}, volume fluctuation")
    #     os.remove(file)

    # check if the dataset contains new enough entries
    elif datetime.datetime.strptime(df["Date"][-1], "%Y-%m-%d 00:00:00+09:00") < datetime.datetime.now() - datetime.timedelta(days=7):

        print(f">{symbol}, too old data")
        os.remove(file)

    else:

        pass

    return

def filter_trust_data(file:str) -> None:

    df = pl.read_csv(file)
    associ_fund_code = file.split("/")[-1].split(".")[0]

    # check if the dataset contains sufficient entries
    if df.shape[0] < 120:

        print(f">{associ_fund_code}, insufficient entries")
        os.remove(file)

    else:

        pass

    return

def main() -> None:

    stock_files = glob.glob("data/stock/*.csv")
    trust_files = glob.glob("data/trust/*.csv")

    print(f"stock files ({len(stock_files)} files):\n==================")
    with ProcessPoolExecutor(max_workers = 18) as executor:
        for file in stock_files: executor.submit(filter_stock_data, file)

    print(f"trust files ({len(trust_files)}) files:\n==================")
    with ProcessPoolExecutor(max_workers = 18) as executor:
        for file in trust_files: executor.submit(filter_trust_data, file)

if __name__ == "__main__":
    main()
