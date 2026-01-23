import yfinance as yf                              # For downloading stock data
import pandas as pd                                # For data manipulation 
from statsmodels.tsa.arima.model import ARIMA      # For time-series forecasting

def get_stock_data(ticker):
    data = yf.download(ticker, period="2y", interval="1d")   # Downloads 2 years of data to have enough "history" to learn patterns
    return data['Close']

def get_7_day_forecast(series): 
    model = ARIMA(series, order=(5,1,0))                   # The 'Engine': We use ARIMA(5,1,0) 
    model_fit = model.fit()                                # 5: looks at last 5 days, 1: handles trends, 0: smooths errors
    
    
   
    forecast_steps = 7                                                 # Forecasting for the next 7 days
    forecast_res = model_fit.get_forecast(steps=forecast_steps)        # Get forecast results
    
    # Extract the 'Mean' (the line) and 'Confidence Intervals' (the shaded area)
    forecast_df = forecast_res.summary_frame(alpha=0.05)               # 95% confidence
    return forecast_df