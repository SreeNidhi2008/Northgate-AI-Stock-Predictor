from pathlib import Path
import yfinance as yf

stock_data = yf.download("AAPL", period="1y", auto_adjust=False)

Path("data").mkdir(exist_ok=True)
stock_data.to_csv("data/aapl_stock_data.csv")

print("Stock data downloaded successfully!")
print(stock_data.head())