!pip install yfinance ta textblob newsapi-python ccxt alpaca-trade-api streamlit pyngrok
!pip install alpha_vantage

# Replace Alpaca with Yahoo Finance for stock data fetching
import yfinance as yf

# Example: Fetch stock data for Nvidia (NVDA)
nvda = yf.download("NVDA", period="1d", interval="1m")

import yfinance as yf
import numpy as np
from textblob import TextBlob
import pandas as pd
from newsapi import NewsApiClient
import ccxt
import streamlit as st
import datetime

# 1. Fetch Market Data using yfinance
def fetch_data(ticker, start="2015-01-01", end=None, interval="1d"):
    if end is None:
        end = pd.to_datetime('today').strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, interval=interval)
    df.dropna(inplace=True)
    return df

btc = fetch_data("BTC-USD")
nvda = fetch_data("NVDA")

# 2. Sentiment Analysis using TextBlob and NewsAPI
def sentiment_score(texts):
    return np.mean([TextBlob(t).sentiment.polarity for t in texts])

# Fetch sentiment using NewsAPI
def fetch_sentiment(ticker="Bitcoin"):
    newsapi = NewsApiClient(api_key='ad16bd71fe124b7bb99ef14e4c355da2')  # Replace with your NewsAPI key
    headlines = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy')
    news_texts = [article['title'] for article in headlines['articles']]
    
    sentiment_scores = [TextBlob(text).sentiment.polarity for text in news_texts]
    return np.mean(sentiment_scores)

btc_sentiment = fetch_sentiment("Bitcoin")
nvda_sentiment = fetch_sentiment("Nvidia")

# 3. Feature Engineering: Technical Indicators (RSI, SMA) and Sentiment
def add_features(df, sentiment=0.0):
    df["Return"] = df["Close"].pct_change()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = (df["Return"].rolling(14).mean()) / (df["Return"].rolling(14).std() + 1e-6)
    df["Sentiment"] = sentiment
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df.dropna(inplace=True)
    return df

btc = add_features(btc, btc_sentiment)
nvda = add_features(nvda, nvda_sentiment)

# 4. Simple Signal Prediction: Buy, Hold, or Sell
def get_trade_signal(ticker="BTC-USD"):
    df = yf.download(ticker, period="1d", interval="1m")
    news_headlines = [
        "Bitcoin adoption rises as ETF demand surges",
        "Nvidia reports record-breaking chip sales",
        "Crypto market faces regulatory uncertainty"
    ]
    sentiment = sentiment_score(news_headlines)
    if sentiment > 0.5:
        signal = "Buy"
    elif sentiment < -0.5:
        signal = "Sell"
    else:
        signal = "Hold"
    
    return signal, df

# 5. Streamlit Web Interface
st.title("AI Trading Signals")

ticker = st.text_input("Enter Asset Symbol", "BTC-USD")

if ticker:
    signal, df = get_trade_signal(ticker)
    st.write(f"Signal: {signal}")
    st.line_chart(df['Close'])
    st.write(f"Sentiment Score: {np.random.uniform(-1, 1)}")  # Replace with actual sentiment score
    st.write(df.tail())

# 6. Trade Execution - Paper Trading Mode (Binance for Crypto)
# Binance API (Crypto Trading)
binance = ccxt.binance()

def execute_trade(action, asset, amount):
    if action == "Buy":
        order = binance.create_market_buy_order(f"{asset}/USDT", amount)
        print(f"Buy Order: {order}")
    elif action == "Sell":
        order = binance.create_market_sell_order(f"{asset}/USDT", amount)
        print(f"Sell Order: {order}")

# Example of executing trade (paper trading mode for now)
amount = 0.1  # Example: buy 0.1 BTC
execute_trade(signal, ticker, amount)


# The rest of the feature engineering
    df.dropna(inplace=True)
    return df

btc = add_features(btc, btc_sentiment)
nvda = add_features(nvda, nvda_sentiment)

# 4. Building the Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Prepare the data
X_btc = btc[['SMA_10', 'SMA_50', 'RSI', 'Sentiment']]
y_btc = btc['Target']
X_nvda = nvda[['SMA_10', 'SMA_50', 'RSI', 'Sentiment']]
y_nvda = nvda['Target']

# Train-test split
X_btc_train, X_btc_test, y_btc_train, y_btc_test = train_test_split(X_btc, y_btc, test_size=0.2, random_state=42)
X_nvda_train, X_nvda_test, y_nvda_train, y_nvda_test = train_test_split(X_nvda, y_nvda, test_size=0.2, random_state=42)

# Train the model
btc_model = RandomForestClassifier(n_estimators=100, random_state=42)
nvda_model = RandomForestClassifier(n_estimators=100, random_state=42)

btc_model.fit(X_btc_train, y_btc_train)
nvda_model.fit(X_nvda_train, y_nvda_train)

# Make predictions
btc_pred = btc_model.predict(X_btc_test)
nvda_pred = nvda_model.predict(X_nvda_test)

# Calculate accuracy
btc_accuracy = accuracy_score(y_btc_test, btc_pred)
nvda_accuracy = accuracy_score(y_nvda_test, nvda_pred)

# 5. Streamlit Application
st.title("Stock Market Prediction App")
st.write("This app predicts whether a stock will go up or down based on sentiment and technical analysis.")

st.header("Market Data")
st.write("### BTC Data", btc.tail())
st.write("### NVDA Data", nvda.tail())

st.header("Sentiment Analysis")
st.write(f"Bitcoin Sentiment Score: {btc_sentiment}")
st.write(f"Nvidia Sentiment Score: {nvda_sentiment}")

st.header("Model Accuracy")
st.write(f"BTC Model Accuracy: {btc_accuracy * 100:.2f}%")
st.write(f"NVDA Model Accuracy: {nvda_accuracy * 100:.2f}%")

# Show prediction results
st.write("### Prediction Results")
btc_predicted = btc_model.predict(X_btc.tail(1)[['SMA_10', 'SMA_50', 'RSI', 'Sentiment']])
nvda_predicted = nvda_model.predict(X_nvda.tail(1)[['SMA_10', 'SMA_50', 'RSI', 'Sentiment']])
st.write(f"BTC Next Day Prediction: {'Up' if btc_predicted[0] == 1 else 'Down'}")
st.write(f"NVDA Next Day Prediction: {'Up' if nvda_predicted[0] == 1 else 'Down'}")

# Optional: Plotting
st.header("Stock Data Visualization")
st.line_chart(btc['Close'], use_container_width=True)
st.line_chart(nvda['Close'], use_container_width=True)

