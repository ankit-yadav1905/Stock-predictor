import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import datetime as dt

# Load model and scaler
model = load_model("lstm_stock_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

st.title("📈 Next Day Stock Price Predictor")

# User input
ticker = st.text_input("Enter stock ticker:", "AAPL")
last_price = st.number_input("Enter last closing price:", min_value=0.0)

if st.button("Predict Next Day Price"):
    last_price_scaled = scaler.transform([[last_price]])
    pred_scaled = model.predict(np.array([last_price_scaled]))
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]

    tomorrow = dt.date.today() + dt.timedelta(days=1)
    st.success(f"Predicted price for {ticker} on {tomorrow} is ${pred_price:.2f}")
