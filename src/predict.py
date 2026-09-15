import joblib
import pandas as pd

model = joblib.load("models/aapl_random_forest.pkl")
data = pd.read_csv("data/aapl_features.csv")

feature_columns = ["Close", "MA_7", "MA_30", "MA_90", "Daily_Return", "Volatility_7", "RSI_14"]
latest_data = data[feature_columns].tail(1)

predicted_price = model.predict(latest_data)[0]
current_price = data["Close"].iloc[-1]

print(f"Latest closing price: ${current_price:.2f}")
print(f"Predicted next closing price: ${predicted_price:.2f}")