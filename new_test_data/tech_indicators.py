import pandas as pd
import ta  # Technical Analysis library
from tqdm import tqdm  # Progress bar

# Load data
df = pd.read_csv("btc_hourly_last_n_hours.csv")

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by time (just in case)
df = df.sort_values(by="timestamp")

### 📊 1. Price-Based Features ###
df["return_1h"] = df["close"].pct_change()  # 1-hour return
df["return_24h"] = df["close"].pct_change(24)  # 24-hour return
df["volatility_24h"] = df["return_1h"].rolling(24).std()  # Rolling volatility

# Moving Averages
df["SMA_50"] = df["close"].rolling(window=50).mean()  # 50-hour SMA
df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()  # 50-hour EMA

### 📈 2. Technical Indicators ###
df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()  # RSI
df["MACD"] = ta.trend.MACD(df["close"]).macd()  # MACD

# Bollinger Bands
bb = ta.volatility.BollingerBands(df["close"], window=20)
df["BB_upper"] = bb.bollinger_hband()
df["BB_lower"] = bb.bollinger_lband()

### 📊 3. Volume-Based Features ###
df["volume_change_1h"] = df["volume"].pct_change()  # Volume % Change
df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()  # OBV

# Drop NaN values from rolling calculations
df.dropna(inplace=True)

# Save processed data
df.to_csv("btc_hourly_last_n_hours_features.csv", index=False)

print(f"✅ Feature engineering complete! Data saved to 'btc_hourly_last_n_hours_features.csv' with {df.shape[1]} columns.")
