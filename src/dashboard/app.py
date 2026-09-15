import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Northgate Stock Predictor", page_icon="📈")

data = pd.read_csv("data/aapl_features.csv")
model = joblib.load("models/aapl_random_forest.pkl")

feature_columns = ["Close", "MA_7", "MA_30", "Daily_Return"]
latest_data = data[feature_columns].tail(1)

current_price = data["Close"].iloc[-1]
predicted_price = model.predict(latest_data)[0]

st.title("📈 Northgate AI Stock Predictor")
st.write("A beginner-friendly dashboard using Apple stock data.")

first_column, second_column = st.columns(2)
first_column.metric("Latest closing price", f"${current_price:.2f}")
second_column.metric("Predicted next closing price", f"${predicted_price:.2f}")

st.subheader("Closing-price history")
chart_data = data[["Date", "Close"]].copy()
chart_data["Date"] = pd.to_datetime(chart_data["Date"])
st.line_chart(chart_data.set_index("Date"))

st.caption("This is an educational machine-learning project, not financial advice.")