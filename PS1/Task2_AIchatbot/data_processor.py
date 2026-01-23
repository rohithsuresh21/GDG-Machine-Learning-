import yfinance as yf    # Importing yfinance library to fetch current and historical financial data
import pandas as pd      # Importing pandas library for data manipulation and analysis
import numpy as np       # importing numpy to perform mathematical callculations on data
from statsmodels.tsa.arima.model import ARIMA   # Importing ARIMA model from statsmodels for time series forecasting
from statsmodels.tsa.stattools import adfuller  # Importing adfuller test for checking stationarity of time series data
from sklearn.metrics import mean_squared_error  # Importing mean_squared_error to evaluate model performance

# data preparation and the pre-filter
def handle_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1                                        # Interquartile range (IQR) cleans extreme noise from data
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return series.clip(lower_bound, upper_bound)        # Clipping noise (outliers) from the data

def check_stationarity(series):
    # Perform Augmented Dickey-Fuller test (ADAF) to check for stationarity
    result = adfuller(series.dropna())
    return result[1]                                    # returning p-value

def get_stock_data(ticker):
    # Downloading 2 years of daily stock data
    data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)
    if data.empty:
        raise ValueError("No data found for the given ticker symbol.")

    close_prices = data['Close'].squeeze()                  # extracting closing prices from the data
    clean_prices = handle_outliers(close_prices)  # Handling outliers in the closing prices

    # Log transform for better stationarity
    clean_prices = np.log(clean_prices)
    return clean_prices

def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))  # Calculating Root Mean Squared Error (RMSE)

def get_7_day_forecast(series):
    # validation split
    train = series[:-7]                           # Using all data except the last 7 days for training
    test = series[-7:]                            # Using the last 7 days for testing/validation

    # check stationarity and decide differencing order
    p_value = check_stationarity(train)
    d = 0 if p_value < 0.05 else 1                 # p-value < 0.05 indicates stationarity

    # validate the model using same parameters
    temp_model = ARIMA(train, order=(5, d, 0)).fit()     # Temporary ARIMA model for validation
    test_predictions = temp_model.forecast(steps=7)     # Forecasting the last 7 days for validation

    # We use np.exp to convert back to original currency for the RMSE
    rmse = calculate_rmse(np.exp(test), np.exp(test_predictions))       # Calculating RMSE for validation

    # train final model on full data
    model = ARIMA(series, order=(5, d, 0))               # Choosing ARIMA parameters based on stationarity
    model_fit = model.fit()                              # Fitting the ARIMA model to the data

    # Forecasting the next 7 days with confidence intervals
    forecast_obj = model_fit.get_forecast(steps=7)

    # Final cleanup: Convert log values back to actual dollar prices
    forecast_mean = np.exp(forecast_obj.predicted_mean)
    conf_int = np.exp(forecast_obj.conf_int())

    return forecast_mean, conf_int, rmse, p_value