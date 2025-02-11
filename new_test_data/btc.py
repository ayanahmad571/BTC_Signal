import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

# Global constant for the number of hours to fetch
HOURS_TO_FETCH = 100

def fetch_historical_ohlcv(symbol="BTC/USDT", timeframe="1h", since=None, limit=1000):
    """
    Fetch historical OHLCV data from Binance API in batches.
    """
    binance = ccxt.binance()
    all_data = []

    # Binance allows max 1000 candles per request, so we loop
    while since < datetime.timestamp(datetime.now()) * 1000:
        try:
            # Fetch data
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break  # Stop if no more data

            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Move to next batch

            # Sleep to avoid rate limits
            time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    return all_data

# Start time (last `HOURS_TO_FETCH` hours)
end_time = datetime.now()
start_time = end_time - timedelta(hours=HOURS_TO_FETCH)  # Subtract `HOURS_TO_FETCH` hours from the current time
start_timestamp = int(start_time.timestamp() * 1000)

# Fetch the data
print(f"Fetching the last {HOURS_TO_FETCH} hours of hourly BTC/USDT data...")
historical_data = fetch_historical_ohlcv(symbol="BTC/USDT", timeframe="1h", since=start_timestamp)

# Convert to DataFrame
df = pd.DataFrame(historical_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# Save to CSV
df.to_csv(f"btc_hourly_last_n_hours.csv", index=False)

print(f"✅ Data collection complete! Saved {len(df)} rows to 'btc_hourly_last_n_hours.csv'.")
