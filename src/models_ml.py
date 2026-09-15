from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("data/aapl_features.csv")

# Tomorrow's closing price is what the model will learn to predict.
data["Tomorrow_Close"] = data["Close"].shift(-1)
data = data.dropna()

feature_columns = ["Close", "MA_7", "MA_30", "Daily_Return"]

X = data[feature_columns]
y = data["Tomorrow_Close"]

# First 80% is for learning; final 20% checks the model.
split_point = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/aapl_random_forest.pkl")

print("Model trained successfully!")
print(f"Average prediction error: ${error:.2f}")