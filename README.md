
This repository contains my submissions for the AI/ML domain inductions. It includes financial time-series forecasting, an AI-powered analytical chatbot, and a scratch implementation of a generative diffusion model.

*PROJECT STRUCTURE:*
/GDG-Inductions-AI
├── PS1/
│   ├── Task1_Forecasting/   # Stock Trend Prediction Engine
│   └── Task2_Chatbot/       # Capital Pulse AI Agent (Streamlit)
├── PS2/
│   └── Task1_Diffusion/     # Mathematical implementation of Diffusion
├── requirements.txt         # Project dependencies
└── README.md                # Submission overview

# PROBLEM STATEMENT 1
## Task 1: Stock Price Forecasting
>>>Description: Built a predictive engine using the ARIMA (AutoRegressive Integrated Moving Average) model to forecast stock prices. The model identifies trends and seasonality in historical data fetched via yfinance.
*Key Logic: Automated p, d, q parameter selection for optimal fit.*
>>>Mathematical Approach: Performed Stationarity checks using the Augmented Dickey-Fuller (ADF) test. Applied Differencing (d) to make the data stationary, then optimized p (lag order) and q (moving average window) using AIC/BIC criteria.
>>>Data Source: Real-time data fetched via yfinance.

## Task 2: Capital Pulse Chatbot
>>>Description: An interactive Streamlit dashboard that combines technical forecasting with fundamental analysis.
*RAG Implementation: Uses Groq (Llama-3) to analyze financial news and provide sentiment-aware investment insights.*
>>>Core Features: Integrates live stock price data with an AI Agent powered by Llama-3 (via Groq LPU).
>>>RAG Logic: The agent scrapes recent financial news and cross-references it with technical indicators (RSI, Moving Averages) to provide a "Buy/Hold/Sell" sentiment analysis.
>>>UI: Built with Streamlit for real-time visualization of price trends and chat interactions.

# PROBLEM STATEMENT 2
## Task 1: Diffusion Model from Scratch
>>>Description: A deep dive into the "mathematics of noise." This task implements the core components of a Denoising Diffusion Probabilistic Model (DDPM).
*Forward Process: Adding Gaussian noise to images over T timesteps.*
*Reverse Process: Training a model to predict the noise and reconstruct the original image.*
>>>1. model.py (The Architecture)
This file contains the "brain" of the operation. In Diffusion models, we typically use a U-Net architecture.
Purpose: It defines the neural network that learns to predict the noise added to an image.
Key Components: It includes Downsampling blocks, Upsampling blocks, and Residual connections. It also must handle Time Embeddings, which tell the model which "step" of the noise process it is currently looking at.
>>>diffusion.py (The Mathematical Logic)
This is the most important file for the "intricate task" evaluation. It contains the mathematical formulas for the diffusion process.
>>>train.py (The Executioner)
This is the file you actually "run" to start the process.
Purpose: It connects the model.py and diffusion.py. It loops through the dataset, calculates the Mean Squared Error (MSE) loss between the predicted noise and the actual noise, and updates the model weights using an optimizer like Adam.
