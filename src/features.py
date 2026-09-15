import pandas as pd

data = pd.read_csv("data/aapl_cleaned_data.csv")

# Moving averages show short-, medium-, and longer-term price trends.
data["MA_7"] = data["Close"].rolling(window=7).mean()
data["MA_30"] = data["Close"].rolling(window=30).mean()
data["MA_90"] = data["Close"].rolling(window=90).mean()

# Daily price movement and recent volatility.
data["Daily_Return"] = data["Close"].pct_change()
data["Volatility_7"] = data["Daily_Return"].rolling(window=7).std()

# RSI helps indicate whether recent price movement is relatively strong or weak.
price_change = data["Close"].diff()
gains = price_change.clip(lower=0)
losses = -price_change.clip(upper=0)

average_gain = gains.rolling(window=14).mean()
average_loss = losses.rolling(window=14).mean()

relative_strength = average_gain / average_loss
data["RSI_14"] = 100 - (100 / (1 + relative_strength))

data = data.dropna()

data.to_csv("data/aapl_features.csv", index=False)

print("Improved features created successfully!")
print(data.head())