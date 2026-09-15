import pandas as pd

data = pd.read_csv("data/aapl_cleaned_data.csv")

data["MA_7"] = data["Close"].rolling(window=7).mean()
data["MA_30"] = data["Close"].rolling(window=30).mean()

data["Daily_Return"] = data["Close"].pct_change()

data = data.dropna()

data.to_csv("data/aapl_features.csv", index=False)

print("Features created successfully!")
print(data.head())