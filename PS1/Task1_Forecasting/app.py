import streamlit as st                                                   # For building the web app    
import matplotlib.pyplot as plt                                          # For plotting
from Data_Processor import get_stock_data, get_7_day_forecast            # Our data processing functions

st.title("Capital Pulse: Financial Intelligence")

# 1. User Input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS):", value="AAPL")

if st.button("Analyze & Forecast"):
    # 2. Get Data
    prices = get_stock_data(ticker)
    
    # 3. Get Forecast
    forecast = get_7_day_forecast(prices)
    
    # 4. Visualize (The part the judges will love)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot last 30 days of real data
    prices.tail(30).plot(ax=ax, label="Historical Price", color="blue")
    
    # Plot the 7-day forecast mean
    forecast['mean'].plot(ax=ax, label="7-Day Forecast", color="red", linestyle="--")
    
    # Plot the Shaded Confidence Interval
    ax.fill_between(forecast.index, 
                    forecast['mean_ci_lower'], 
                    forecast['mean_ci_upper'], 
                    color='red', alpha=0.2, label="95% Confidence")
    
    plt.legend()
    st.pyplot(fig)