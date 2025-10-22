"""
Fetch historical stock data using yfinance and save as CSV files under src/data/yahoo-finance/.
"""

import os
from datetime import datetime

import yfinance as yf

# Define stock symbols (extend as needed)
STOCKS = {
    "apple": "AAPL",
    "google": "GOOGL",
    "meta": "META",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "intel": "INTC",
}

# Configure output folder
DATA_DIR = os.path.join("src", "data", "yahoo-finance")
os.makedirs(DATA_DIR, exist_ok=True)

# Define date range
START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


def fetch_and_save_stock(symbol: str, name: str, start: str, end: str):
    """
    Fetch historical stock data for a given symbol and save it to CSV.
    """
    print(f"📈 Fetching {name} ({symbol}) from {start} to {end} ...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start, end=end)

    if data.empty:
        print(f"⚠️ No data found for {symbol}. Skipping.")
        return

    file_path = os.path.join(DATA_DIR, f"{name.lower()}.csv")
    data.to_csv(file_path, index=True)
    print(f"✅ Saved: {file_path} ({len(data)} rows)")


def main():
    for name, symbol in STOCKS.items():
        fetch_and_save_stock(symbol, name, START_DATE, END_DATE)
    print(f"\n🎉 All stock data saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
