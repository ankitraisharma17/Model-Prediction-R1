import streamlit as st
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model

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
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(window_size, 4)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss="mse", optimizer="adam")
    return model

# Load trained models
btc_model = load_model("bitcoin_model.h5")
sol_model = load_model("solana_model.h5")

# Streamlit UI
st.title("Cryptocurrency Price Prediction")
coin = st.selectbox("Select a cryptocurrency", ["Bitcoin", "Solana"])
days = st.slider("Select the number of days for data", 30, 365, 365)

if st.button("Fetch Data"):
    st.write(f"Fetching {coin} data for the last {days} days...")

    # Fetch data
    data = fetch_crypto_data(coin.lower(), days=str(days))
    if data is not None:
        st.line_chart(data.set_index('timestamp')['price'])
        st.write("Data fetched successfully!")

        # Compute indicators
        data = compute_indicators(data)

        # Prepare data for LSTM
        X, y = prepare_data(data)
        
        # Make predictions with the selected model
        model = btc_model if coin.lower() == "bitcoin" else sol_model
        predictions = model.predict(X)
        
        # Show predictions
        st.write(f"Predicted {coin} price for the next day: {predictions[-1][0]}")
        
        # Plot predictions
        st.line_chart(predictions.flatten())
