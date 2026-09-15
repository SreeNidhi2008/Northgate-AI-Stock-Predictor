import pandas as pd

data = pd.read_csv("data/aapl_stock_data.csv", skiprows=[1, 2])

data = data.rename(columns={"Price": "Date"})
data["Date"] = pd.to_datetime(data["Date"])

data = data.dropna()
data = data.sort_values("Date")

data.to_csv("data/aapl_cleaned_data.csv", index=False)

print("Cleaned data saved successfully!")
print(data.head())