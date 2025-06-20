#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Tesla.csv", encoding='ISO-8859-1')

# Clean column names
df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=False)

# Convert 'Date' to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Rename and select relevant columns
df = df[['Date', 'Close']].rename(columns={'Close': 'Close_Price'})

# Convert Close_Price to numeric (handles any commas or strings)
df['Close_Price'] = pd.to_numeric(df['Close_Price'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['Close_Price'])

# Convert Date to ordinal for regression
df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

# Prepare features and labels
X = df[['Date_ordinal']]
y = df['Close_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], y, label='Actual Prices', color='blue')
plt.plot(df['Date'].iloc[len(X_train):], predictions, label='Predicted Prices', color='red')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Tesla Share Price Forecasting (Linear Regression)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




