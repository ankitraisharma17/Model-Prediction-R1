import streamlit as st
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Function to fetch cryptocurrency data
def fetch_crypto_data(coin="bitcoin", days="365"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
    headers = {"User-Agent": "Mozilla/5.0"}  # Avoid blocking

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return None

# Function to compute technical indicators (SMA, EMA, RSI)
def compute_indicators(df):
    df["SMA_10"] = df["price"].rolling(window=10).mean()
    df["SMA_50"] = df["price"].rolling(window=50).mean()
    df["EMA_10"] = df["price"].ewm(span=10, adjust=False).mean()
    df["RSI"] = compute_rsi(df["price"])
    return df

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Prepare data for LSTM model
def prepare_data(df, window_size=10):
    data = []
    labels = []
    for i in range(len(df) - window_size):
        data.append(df.iloc[i:i+window_size, 1:].values)
        labels.append(df.iloc[i+window_size]["price"])
    return np.array(data), np.array(labels)

# Function to build LSTM model
def build_lstm_model(window_size):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 4)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(loss="mse", optimizer=Adam())
    return model

# St
