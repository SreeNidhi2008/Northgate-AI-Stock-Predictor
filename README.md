# Northgate AI Stock Predictor

A beginner-friendly machine-learning project that downloads Apple stock data, prepares it, trains a prediction model, and displays the result in a Streamlit dashboard.

## Features

- Downloads one year of Apple stock data using yfinance
- Cleans and prepares the data
- Creates moving-average and daily-return features
- Trains a Random Forest model
- Predicts the next closing price
- Displays results in a Streamlit dashboard

## Run the dashboard

```powershell
streamlit run src\dashboard\app.py