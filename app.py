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

# Streamlit UI
st.title("Ankits Prediction Model")

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
        window_size = 10
        X, y = prepare_data(data, window_size)

        # Build and train the LSTM model
        model = build_lstm_model(window_size)
        model.fit(X, y, epochs=5, batch_size=32)

        # Make predictions for each day (not just the next day)
        predictions = model.predict(X)

        # Display predictions alongside the actual prices
        prediction_df = pd.DataFrame({
            "timestamp": data["timestamp"].iloc[window_size:].values,
            "actual_price": data["price"].iloc[window_size:].values,
            "predicted_price": predictions.flatten()
        })

        st.write("Predictions vs Actual Prices")
        st.line_chart(prediction_df.set_index('timestamp')[['actual_price', 'predicted_price']])

        st.write(f"Predicted prices for {coin} for the next {days} days:")
        st.write(prediction_df)
