import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Fetch historical cryptocurrency data
def fetch_crypto_data(coin="bitcoin", days="365"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

# Prepare data for LSTM
def prepare_data(df, window_size=60):
    data = df[['price']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data_scaled[i-window_size:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y, scaler

# Build LSTM model
def build_lstm_model(X):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Predict next price
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Train the model
def train_model(coin="bitcoin", days="365"):
    print(f"Fetching {coin} data...")
    data = fetch_crypto_data(coin, days)

    if data is not None:
        X, y, scaler = prepare_data(data)

        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = build_lstm_model(X_train)

        print("Training the model...")
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        print(f"Saving {coin} model...")
        model.save(f"{coin}_model.h5")

        print(f"{coin} model saved successfully!")

# Train models for Bitcoin and Solana
train_model("bitcoin", "365")
train_model("solana", "365")
