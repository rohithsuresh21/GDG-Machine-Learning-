import streamlit as st                                  # streamlit library for web dashboard creation
import numpy as np                                      # numpy for mathematical calculations
import matplotlib.pyplot as plt                         # matplotlib for data visualization
import pandas as pd                                     # pandas for data manipulation
from groq import Groq                                   # groq library to interact with Groq LPU API
from data_processor import get_stock_data, get_7_day_forecast

# PAGE CONFIGURATION
st.set_page_config(page_title="CAPITAL PULSE", layout="wide")
st.title("STOCK FORECASTING CHATBOT")

#groq client setup
try:
    # Use the LABEL from your secrets file, not the key itself
    GROQ_API_KEY = st.secrets["groq"]["api_key"]                         # Fetching Groq API key from Streamlit secrets
    groq_key = GROQ_API_KEY                                             
    client = Groq(api_key=groq_key)
    MODEL_NAME = "llama-3.3-70b-versatile"                               # Specifying the Groq LPU model to use
except Exception as e:
    st.error(":( API Key Error: Please make sure GROQ_API_KEY is in your .streamlit/secrets.toml file.")
    st.stop()

# SESSION STATE
if "data_loaded" not in st.session_state:              
    st.session_state.data_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# SIDEBAR
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Ticker", value="AAPL").upper()                # Text input for stock ticker symbol
run_button = st.sidebar.button("Run Forecast")

if run_button and ticker:
    try:
        series = get_stock_data(ticker)
        f_mean, f_conf, rmse, p_val = get_7_day_forecast(series)                 # Getting 7-day forecast and model metrics
        st.session_state.update({
            "series": series, "f_mean": f_mean, "f_conf": f_conf,
            "rmse": rmse, "p_val": p_val, "data_loaded": True, "ticker": ticker
        })
    except Exception as e:
        st.error(f"Error: {e}")

# DASHBOARD DISPLAY 
if st.session_state.data_loaded:
    hist_prices = np.exp(st.session_state.series)
    # Generate 7 business days starting from today
    forecast_dates = pd.date_range(start=pd.Timestamp.now().normalize(), periods=7, freq="B")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 4))
        recent = hist_prices.tail(15)                                        # Zoom in on the last 15 days
        ax.plot(recent.index, recent.values, label='History', marker='o')
        ax.plot(forecast_dates, st.session_state.f_mean, label='Forecast', color='orange', ls='--', marker='o')
        plt.xticks(rotation=45)
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Model Metrics")
        st.metric("Validation RMSE", f"${st.session_state.rmse:.2f}")
        st.write(f"Stationary: {'Yes' if st.session_state.p_val < 0.05 else 'No'}")
        # Table of forecasted prices
        st.dataframe(pd.DataFrame(st.session_state.f_mean).rename(columns={0:"Price"}).style.format("${:.2f}"))

# chat interface 
st.divider()
st.subheader(" AI Financial Assistant ")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask about the trend, RMSE, or stationarity..."):        # Chat input box for user queries
    # Display user message in chat
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Adding Data Context (This is how the AI "sees" your chart data)
    context = "No data loaded."
    if st.session_state.data_loaded:
        last_p = np.exp(st.session_state.series.iloc[-1])
        context = (f"Stock: {st.session_state.ticker}. Current: {last_p:.2f}. "
                   f"7-Day Forecast: {st.session_state.f_mean.values.tolist()}. "
                   f"Model RMSE: {st.session_state.rmse:.2f}.")

    with st.chat_message("assistant"):
        try:
            # Groq API Call
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": f"You are a professional stock analyst. Context: {context}"},
                    {"role": "user", "content": prompt}
                ]
            )
            response = completion.choices[0].message.content
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Groq API Error: {e}")


