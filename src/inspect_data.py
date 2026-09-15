import pandas as pd

data = pd.read_csv("data/aapl_stock_data.csv")

print("Number of rows and columns:", data.shape)
print("\nColumn names:")
print(data.columns)

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())