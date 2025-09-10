import yfinance as yf
import asyncio
import requests
import time

async def _yf_download(ticker: str, download_dir:str) -> None:

    try:

        stock_data = yf.Ticker(ticker).history(interval="1d", auto_adjust=True, period="max")
        
        if len(stock_data) == 0:

            return
        
        stock_data.to_csv(f"{download_dir}/{ticker}.csv")
        
        return
    
    except Exception as e:

        print(f"Error downloading {ticker}: {e}")
        return

async def _trust_download(associ_fund_code: str, download_dir:str) -> None:

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

    
    time.sleep(1)
    return

async def _runner(f):
    def _wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return _wrapper

@_runner
async def download_data(asset_class:str, ticker_list:list[str], download_dir:str) -> None:

    if asset_class == "stock": tasks = [_yf_download(ticker, download_dir) for ticker in ticker_list]
    elif asset_class == "trust": tasks = [_trust_download(code, download_dir) for code in ticker_list]
    else:
        raise ValueError("Unsupported asset class. Use 'stock' or 'trust'.")
    
    await asyncio.gather(*tasks)

    return
