import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Streamlit Page Config
st.set_page_config(page_title="Stock Price Prediction with LSTM", layout="wide")

# Title
st.title("📈 Stock Price Prediction using LSTM")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file, header=0, index_col=0)
    
    # Clean data
    if "Volume" in df.columns:
        df.drop(['Volume'], axis=1, inplace=True)
    if "Date" in df.columns:
        df.drop(['Date'], axis=1, inplace=True)
    df.dropna(inplace=True)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))

    # Plot Close price
    st.subheader("📉 Close Price History")
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df['Close'], label="Close Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Data Preprocessing
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("⏳ Training the LSTM model..."):
        history = model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=0)

    st.success("✅ Model training complete!")

    # Create Testing Data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # RMSE
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    # Results DataFrame
    train = data[:training_data_len]
    valid = data[training_data_len:].copy()
    valid['Predictions'] = predictions

    # Display Metrics
    st.metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse:.2f}")

    # Plot Predictions
    st.subheader("📊 Model Predictions vs Actual")
    fig2, ax2 = plt.subplots(figsize=(14,6))
    ax2.plot(train['Close'], label="Training Data")
    ax2.plot(valid['Close'], label="Actual Price")
    ax2.plot(valid['Predictions'], label="Predicted Price")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Close Price")
    ax2.legend()
    st.pyplot(fig2)

    # Show table with predictions
    st.subheader("🔎 Prediction Table")
    st.dataframe(valid.tail(20))

else:
    st.info("👆 Please upload a CSV file to start.")
