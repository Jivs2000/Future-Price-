#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(layout="centered")
st.title("📈 Tesla Share Price Prediction")
st.markdown("This app uses **Linear Regression** to forecast future Tesla stock prices.")

# File upload
uploaded_file = st.file_uploader("Upload Tesla stock CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

        # Clean column names
        df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=False)

        # Convert 'Date' column
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Filter and clean necessary columns
        df = df[['Date', 'Close']].rename(columns={'Close': 'Close_Price'})
        df['Close_Price'] = pd.to_numeric(df['Close_Price'], errors='coerce')
        df = df.dropna(subset=['Close_Price'])

        # Convert date to ordinal
        df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

        # Train-test split
        X = df[['Date_ordinal']]
        y = df['Close_Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Future prediction
        future_days = st.slider("Predict how many days into the future?", 30, 365, 90)
        last_date = df['Date'].max()
        future_dates = pd.date_range(start=last_date, periods=future_days+1, freq='D')[1:]
        future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        future_predictions = model.predict(future_ordinals)

     
        # Display forecast data
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_predictions})
        st.subheader("📊 Forecasted Prices")
        st.dataframe(forecast_df.set_index('Date').round(2))

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Upload a Tesla stock CSV file to begin.")


# In[ ]:




