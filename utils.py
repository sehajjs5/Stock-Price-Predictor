import joblib
import os
import numpy as np
import yfinance as yf
import pandas as pd

MODELS_DIR = "models"

def get_model(ticker):
    path = os.path.join(MODELS_DIR, f"{ticker}.pkl")
    if not os.path.exists(path):
        raise ValueError("Model not found for selected stock")
    return joblib.load(path)


def get_latest_features(ticker):
    df = yf.download(ticker, period="3mo")

    df["Close_lag_1"] = df["Close"].shift(1)
    df["Close_lag_5"] = df["Close"].shift(5)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(5).std()

    df = df.dropna()
    latest = df.iloc[-1]

    features = [
        latest["Open"],
        latest["Volume"],
        latest["Close_lag_1"],
        latest["Close_lag_5"],
        latest["MA_5"],
        latest["MA_10"],
        latest["Volatility"]
    ]

    return np.array(features).reshape(1, -1), df

def get_latest_features_with_user_input(ticker, user_open, user_volume):
    df = yf.download(ticker, period="3mo")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # --- Feature Engineering ---
    df["Close_lag_1"] = df["Close"].shift(1)
    df["Close_lag_5"] = df["Close"].shift(5)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(5).std()

    df = df.dropna()

    latest = df.iloc[-1]

    # Replace Open & Volume with user inputs
    features = [
        float(user_open),
        float(user_volume),
        float(latest["Close_lag_1"]),
        float(latest["Close_lag_5"]),
        float(latest["MA_5"]),
        float(latest["MA_10"]),
        float(latest["Volatility"])
    ]

    return np.array(features, dtype=float).reshape(1, -1), df

def predict_stock(ticker):
    model = get_model(ticker)
    X, df = get_latest_features(ticker)

    prediction = model.predict(X)[0]
    close_prices = df["Close"].squeeze() 
    history = {
        "dates": df.index.strftime("%Y-%m-%d").tolist()[-30:] + ["Next Day"],
        "prices": close_prices.iloc[-30:].tolist() + [float(prediction)]
    }

    return round(float(prediction), 2), history
def predict_custom(ticker, user_open, user_volume):
    model = get_model(ticker)
    X, df = get_latest_features_with_user_input(ticker, user_open, user_volume)

    prediction = model.predict(X)[0]
    close_prices = df["Close"].squeeze() 
    history = {
        "dates": df.index.strftime("%Y-%m-%d").tolist()[-30:] + ["Simulated"],
        "prices": close_prices.iloc[-30:].tolist() + [float(prediction)]
    }

    return round(float(prediction), 2), history
