import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import mplfinance as mpf

# Extend the date range up to November
np.random.seed(0)
n_samples = 306  # Extending the date range to November
dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')  # Extend the date range to November
features = np.random.rand(n_samples, 3)
close_price = np.cumsum(np.random.randn(n_samples)) + 100

# Create a DataFrame
data = pd.DataFrame({'Date': dates, 'Open': close_price - 2, 'High': close_price + 1, 'Low': close_price - 4, 'Close': close_price})
data.set_index('Date', inplace=True)

# Feature selection
features = ['Open', 'High', 'Low', 'Close']
X = data[features]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the entire dataset
y_pred = model.predict(X)

# Create a candlestick chart
ohlc = data[['Open', 'High', 'Low', 'Close']]
ohlc.index.name = 'Date'

# Plot the candlestick chart with the line plot for predictions
mpf.plot(ohlc, type='candle', style='yahoo', title="Stock Price Prediction - Kamasani Giridhar Reddy", ylabel='Price',
         figratio=(10, 5), addplot=[mpf.make_addplot(y_pred, color='r')])

# Show the plot

