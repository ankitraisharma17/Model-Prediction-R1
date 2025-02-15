#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas numpy requests scikit-learn matplotlib seaborn tensorflow keras streamlit


# In[4]:


def fetch_crypto_data(coin="bitcoin", days="365"):  # Change 730 to 365
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
    headers = {"User-Agent": "Mozilla/5.0"}  # Avoid blocking

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    else:
        print(f"Error fetching data: {response.status_code}, {response.text}")
        return None

# Fetch Bitcoin and Solana data for the last 365 days
bitcoin_data = fetch_crypto_data("bitcoin", days="365")
solana_data = fetch_crypto_data("solana", days="365")

# Save to CSV
if bitcoin_data is not None:
    bitcoin_data.to_csv("bitcoin_data.csv", index=False)
    print("Bitcoin data saved successfully!")

if solana_data is not None:
    solana_data.to_csv("solana_data.csv", index=False)
    print("Solana data saved successfully!")


# In[5]:


python fetch_data.py


# In[6]:


import pandas as pd
import numpy as np

def compute_indicators(df):
    df["SMA_10"] = df["price"].rolling(window=10).mean()  # Simple Moving Average (10-day)
    df["SMA_50"] = df["price"].rolling(window=50).mean()  # SMA (50-day)
    df["EMA_10"] = df["price"].ewm(span=10, adjust=False).mean()  # Exponential Moving Average
    df["RSI"] = compute_rsi(df["price"])
    return df

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Load data
bitcoin_df = pd.read_csv("bitcoin_data.csv")
solana_df = pd.read_csv("solana_data.csv")

# Compute indicators
bitcoin_df = compute_indicators(bitcoin_df)
solana_df = compute_indicators(solana_df)

# Save processed data
bitcoin_df.to_csv("bitcoin_preprocessed.csv", index=False)
solana_df.to_csv("solana_preprocessed.csv", index=False)

print("Preprocessed data saved!")


# In[7]:


python preprocess_data.py


# In[8]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_data(df, window_size=10):
    data = []
    labels = []
    for i in range(len(df) - window_size):
        data.append(df.iloc[i:i+window_size, 1:].values)
        labels.append(df.iloc[i+window_size]["price"])
    return np.array(data), np.array(labels)

# Load preprocessed data
bitcoin_df = pd.read_csv("bitcoin_preprocessed.csv")
solana_df = pd.read_csv("solana_preprocessed.csv")

# Prepare training data
window_size = 10
X_btc, y_btc = prepare_data(bitcoin_df, window_size)
X_sol, y_sol = prepare_data(solana_df, window_size)

# Build LSTM Model
def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, X_btc.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(loss="mse", optimizer="adam")
    return model

# Train models
btc_model = build_lstm_model()
btc_model.fit(X_btc, y_btc, epochs=20, batch_size=16)

sol_model = build_lstm_model()
sol_model.fit(X_sol, y_sol, epochs=20, batch_size=16)

# Save models
btc_model.save("bitcoin_model.h5")
sol_model.save("solana_model.h5")

print("Models trained and saved!")


# In[10]:


import tensorflow as tf

# Define the loss function
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load the models with custom objects
btc_model = tf.keras.models.load_model("bitcoin_model.h5", custom_objects=custom_objects)
sol_model = tf.keras.models.load_model("solana_model.h5", custom_objects=custom_objects)

