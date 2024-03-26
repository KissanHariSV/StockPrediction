# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load historical SPY data (assuming you have a CSV file with historical data)
spy_data = pd.read_csv('/Users/kissanvenugopal/Downloads/Download Data - FUND_US_ARCX_SPY.csv')  # Update 'spy_data.csv' with your dataset filename

# Data Preprocessing
# Clean the data, handle missing values, normalize features as needed
# Remove commas and convert 'Volume' column to numerical format
spy_data['Volume'] = spy_data['Volume'].str.replace(',', '').astype(float)

# Feature Selection/Engineering
features = ['Open', 'High', 'Low', 'Volume']
X = spy_data[features]
y = spy_data['Close']  # Predicting closing price
y_high = spy_data['High']  # Predicting high price
y_low = spy_data['Low']    # Predicting low price

# Shift the target variable (Close price) by one day to align with previous day's data
y = y.shift(-1)

# Shift the target variables (High and Low prices) by one day to align with previous day's data
y_high = y_high.shift(-1)
y_low = y_low.shift(-1)

# Drop the last row since we don't have data for the next day's performance
X = X[:-1]
y = y[:-1]
y_high = y_high[:-1]
y_low = y_low[:-1]

# Split Data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_high_train, y_high_test, y_low_train, y_low_test = train_test_split(X, y_high, y_low, test_size=0.2, random_state=42)

# Choose Model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_high = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_low = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train Model
model.fit(X_train, y_train)
model_high.fit(X_train, y_high_train)
model_low.fit(X_train, y_low_train)

# Evaluate Model
train_predictions = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
print(f'Training RMSE: {train_rmse}')

test_predictions = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print(f'Testing RMSE: {test_rmse}')


train_predictions_high = model_high.predict(X_train)
train_rmse_high = np.sqrt(mean_squared_error(y_high_train, train_predictions_high))
print(f'Training RMSE for High Price: {train_rmse_high}')

test_predictions_high = model_high.predict(X_test)
test_rmse_high = np.sqrt(mean_squared_error(y_high_test, test_predictions_high))
print(f'Testing RMSE for High Price: {test_rmse_high}')

train_predictions_low = model_low.predict(X_train)
train_rmse_low = np.sqrt(mean_squared_error(y_low_train, train_predictions_low))
print(f'Training RMSE for Low Price: {train_rmse_low}')

test_predictions_low = model_low.predict(X_test)
test_rmse_low = np.sqrt(mean_squared_error(y_low_test, test_predictions_low))
print(f'Testing RMSE for Low Price: {test_rmse_low}')



latest_data = np.array([[latest_open_price, latest_high_price, latest_low_price, latest_volume]])
next_day_performance_prediction = model.predict(latest_data)
print(f'Predicted SPY performance for the next day: {next_day_performance_prediction[0]}')

latest_data = np.array([[latest_open_price, latest_high_price, latest_low_price, latest_volume]])

next_day_high_price = model_high.predict(latest_data)
next_day_low_price = model_low.predict(latest_data)

print(f'Predicted SPY high price for the next day: {next_day_high_price[0]}')
print(f'Predicted SPY low price for the next day: {next_day_low_price[0]}')
