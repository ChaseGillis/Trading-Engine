import pandas as pd
import requests
import csv
import os
from pathlib import Path
from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest

# Function to read API keys from a file
def read_keys(path):
    with open(path, 'r') as f:
        key_id = f.readline().strip()
        secret_key = f.readline().strip()
    return key_id, secret_key

# Main function to fetch data and write to CSV
def get_the_data():
    years = 2
    pd.set_option('display.max_rows', 15)
    pd.set_option('display.max_columns', 10)

    # Access the API keys from environment variables
    key_id = os.environ['KEY_ID']
    secret_key = os.environ['SECRET_KEY']

    # Initialize Alpaca API
    client = StockHistoricalDataClient(key_id, secret_key)

    request_params = StockBarsRequest(
                            symbol_or_symbols = ["TSLA", "AAPL", "GOOG", "MSFT", "T", "TGT", "INTC", "KO", "PEP",
          "AMZN", "NFLX", "FB", "NVDA", "DIS", "AMD", "CRM", "PYPL", "UBER",
          "BABA", "PFE", "MRNA", "JNJ", "NKE", "WMT", "CSCO", "MCD", "SBUX",
          "BA", "V", "MA", "XOM", "CVX", "PG", "WFC", "GS", "JPM", "BAC",
          "SHOP", "SPOT", "TWTR", "SNAP", "SQ", "ZM", "ROKU", "UBER", "LYFT",
          "WORK", "DDOG", "NET", "CRWD", "PTON", "DKNG", "FSLY", "OKTA",
          "ENPH", "PLUG", "CRSP", "SQ", "ZM", "ROKU", "MELI", "PINS", "DOCU",
          "NOW", "ROKU", "SPCE", "RCL", "CCL", "LUV", "UAL", "DAL", "AAL",
          "JBLU", "UAL", "DAL", "CCL", "LUV", "AAL", "JBLU", "SAVE", "ALGT",
          "UAL", "RJET", "ALGT", "SAVE", "LUV", "AAL", "UAL", "RJET", "ALGT"],
                            timeframe=TimeFrame.Day,
                            start=(datetime.now() - timedelta(days=(365*years))).strftime('%Y-%m-%d'),
                            end_date=datetime.now().strftime('%Y-%m-%d')
                            )

    bars = client.get_stock_bars(request_params).df

    return bars

def pull_data():
    bars = get_the_data()
    bars = bars['close']
    bars.to_csv(r"stock_prices.csv")
