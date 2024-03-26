## Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load historical SPY data (assuming you have a CSV file with historical data)
spy_data = pd.read_csv('/Users/kissanvenugopal/Downloads/Download Data - FUND_US_ARCX_SPY.csv')  # Update 'spy_data.csv' with your dataset filename

# Data Preprocessing
# Assuming your dataset has columns like 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', etc.
# You may need to clean the data, handle missing values, and engineer features as needed.
spy_data['Volume'] = spy_data['Volume'].str.replace(',', '').astype(float)
# Feature Selection/Engineering
# For simplicity, let's use only 'Open', 'High', 'Low', 'Volume' as features
features = ['Open', 'High', 'Low', 'Volume']
X = spy_data[features]
y = spy_data['Close']  # Predicting closing price

# Split Data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Evaluate Model
train_predictions = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
print(f'Training RMSE: {train_rmse}')

test_predictions = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print(f'Testing RMSE: {test_rmse}')

# Prediction for the next day
# Assuming you have the latest data available for the next day
# Sample values for the latest data
latest_open_price = 521.11
latest_high_price = 522.61
latest_low_price = 520.97
latest_volume = 79070844  # Assuming this is an integer value

# Prediction for the next day
latest_data = np.array([[latest_open_price, latest_high_price, latest_low_price, latest_volume]])
next_day_prediction = model.predict(latest_data)
print(f'Predicted SPY price for the next day: {next_day_prediction[0]}')

