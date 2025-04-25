from binance.client import Client
import pandas as pd
import os
import time
from datetime import datetime

# Ініціалізація клієнта Binance (без API ключів для public data)
client = Client()

def get_all_klines(symbol="BTCUSDT", interval="15m", start_date="1 Jan, 2022"):
    filename = f"data/{symbol}_15m.csv"
    os.makedirs("data", exist_ok=True)

    start_ts = int(datetime.strptime(start_date, "%d %b, %Y").timestamp() * 1000)
    all_data = []
    limit = 1000

    while True:
        klines = client.get_klines(symbol=symbol, interval=interval, startTime=start_ts, limit=limit)
        if not klines:
            break
        all_data.extend(klines)
        start_ts = klines[-1][0] + 1
        print(f"{symbol}: Collected {len(all_data)} rows so far...")
        time.sleep(0.5)  # пауза, щоб не вбити API

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    df.to_csv(filename, index=False)
    print(f"Saved {symbol} to {filename}")

def download_top_coins():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    for symbol in symbols:
        get_all_klines(symbol)

if __name__ == "__main__":
    download_top_coins()
