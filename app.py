import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("📈 Stock Market Price Predictor")
st.write("Enter a stock ticker symbol (e.g., AAPL, TSLA, RELIANCE.NS) to view predictions")

# Input
ticker = st.text_input("Enter stock ticker:", value="AAPL")

if st.button("Predict"):
    data = yf.download(ticker, start="2020-01-01", end="2024-12-31")

    if data.empty:
        st.error("No data found for this ticker!")
    else:
        df = data[['Close']].copy()
        df['Prediction'] = df[['Close']].shift(-1)
        df.dropna(inplace=True)

        X = df[['Close']]
        y = df['Prediction']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Show plot
        st.subheader(f"{ticker} Predicted vs Actual Closing Prices")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_test.values, label="Actual")
        ax.plot(predictions, label="Predicted")
        ax.legend()
        st.pyplot(fig)
