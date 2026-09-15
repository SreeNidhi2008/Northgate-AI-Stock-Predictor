import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Northgate Stock Predictor", page_icon="📈")
st.title("📈 Northgate AI Stock Predictor")
st.write("Choose a stock to download data and generate an educational prediction.")

stock_options = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
}

selected_name = st.selectbox("Choose a stock", list(stock_options.keys()))
ticker = stock_options[selected_name]

data = yf.download(ticker, period="1y", auto_adjust=False)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.reset_index()

data["MA_7"] = data["Close"].rolling(7).mean()
data["MA_30"] = data["Close"].rolling(30).mean()
data["MA_90"] = data["Close"].rolling(90).mean()
data["Daily_Return"] = data["Close"].pct_change()
data["Volatility_7"] = data["Daily_Return"].rolling(7).std()

price_change = data["Close"].diff()
average_gain = price_change.clip(lower=0).rolling(14).mean()
average_loss = (-price_change.clip(upper=0)).rolling(14).mean()
relative_strength = average_gain / average_loss
data["RSI_14"] = 100 - (100 / (1 + relative_strength))

data["Tomorrow_Close"] = data["Close"].shift(-1)
data = data.dropna()

feature_columns = [
    "Close", "MA_7", "MA_30", "MA_90",
    "Daily_Return", "Volatility_7", "RSI_14",
]

X = data[feature_columns]
y = data["Tomorrow_Close"]

split_point = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predicted_price = model.predict(X.tail(1))[0]
current_price = data["Close"].iloc[-1]
error = mean_absolute_error(y_test, model.predict(X_test))

first_column, second_column, third_column = st.columns(3)
first_column.metric("Latest closing price", f"${current_price:.2f}")
second_column.metric("Predicted next closing price", f"${predicted_price:.2f}")
third_column.metric("Average test error", f"${error:.2f}")

st.subheader(f"{selected_name} closing-price history")
st.line_chart(data.set_index("Date")["Close"])
st.caption("Educational project only — not financial advice.")