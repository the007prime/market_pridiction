from datetime import date, datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import tensorflow as tf
from tensorflow.keras import callbacks, models
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import yfinance as yf

st.set_page_config(
    page_title="Universal Stock Price Predictor", page_icon="📈", layout="wide"
)

st.title("📈 Universal Stock Analysis & LSTM Prediction Dashboard")
st.markdown(
    "Enter any valid stock ticker below to pull live market data, view"
    " technical indicators, and train a custom LSTM model on-the-fly."
)

st.sidebar.header("Market Parameters")

ticker = (
    st.sidebar.text_input(
        "Stock Ticker Symbol",
        value="GOOG",
        help="Try AAPL, MSFT, TSLA, NVDA, AMZN, etc.",
    )
    .upper()
    .strip()
)

start_date = st.sidebar.date_input("Historical Start Date", value=datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

st.sidebar.subheader("Model Hyperparameters")
lookback = st.sidebar.slider(
    "Lookback Window (Days)", min_value=30, max_value=90, value=60
)
epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=50, value=15)
batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64], index=1)


@st.cache_data(ttl=3600)
def fetch_data(symbol, start, end):
  try:
    # Adding auto_adjust=True helps stabilize downloads on cloud IPs
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
      return None

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.droplevel("Ticker")

    df = df.reset_index()

    if "Close" not in df.columns:
      return None

    df["MA100"] = df["Close"].rolling(100).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    return df
  except Exception:
    return None

with st.spinner(f"Fetching market data for {ticker}..."):
  df = fetch_data(ticker, start_date, end_date)

if df is None or len(df) <= (lookback + 20):
  st.error(
      f"Could not retrieve enough data for '{ticker}'. Yahoo Finance might be"
      " temporarily blocking requests from this cloud region, or the ticker"
      " symbol is invalid. Try a different ticker like GOOG or AAPL."
  )
  st.stop()

st.success(
    f"Successfully loaded {len(df)} trading records for **{ticker}**!"
)

with st.expander("View Raw Data Sample"):
  st.dataframe(df.tail(10), use_container_width=True)

st.subheader(f"Price Charts for {ticker}")
tab1, tab2 = st.tabs(
    ["Candlestick Chart", "Moving Averages (MA100 & MA200)"]
)

with tab1:
  fig_candlestick = go.Figure(
      data=[
          go.Candlestick(
              x=df["Date"],
              open=df["Open"],
              high=df["High"],
              low=df["Low"],
              close=df["Close"],
              name=ticker,
          )
      ]
  )
  fig_candlestick.update_layout(
      xaxis_rangeslider_visible=False,
      margin=dict(l=20, r=20, t=30, b=20),
      height=480,
  )
  st.plotly_chart(fig_candlestick, use_container_width=True)

with tab2:
  fig_ma = go.Figure()
  fig_ma.add_trace(
      go.Scatter(
          x=df["Date"],
          y=df["Close"],
          mode="lines",
          name="Close Price",
          line=dict(color="#38BDF8", width=1.5),
      )
  )
  fig_ma.add_trace(
      go.Scatter(
          x=df["Date"],
          y=df["MA100"],
          mode="lines",
          name="100-Day MA",
          line=dict(color="#F59E0B", width=1.5),
      )
  )
  fig_ma.add_trace(
      go.Scatter(
          x=df["Date"],
          y=df["MA200"],
          mode="lines",
          name="200-Day MA",
          line=dict(color="#10B981", width=1.5),
      )
  )
  fig_ma.update_layout(
      xaxis_title="Date",
      yaxis_title="Price (USD)",
      margin=dict(l=20, r=20, t=30, b=20),
      height=480,
  )
  st.plotly_chart(fig_ma, use_container_width=True)

st.subheader("🤖 Dynamic LSTM Prediction Engine")

if st.button(f"Run Prediction Model for {ticker}", type="primary"):
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(df[["Close"]].values)

  X, y = [], []
  for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback : i, 0])
    y.append(scaled_data[i, 0])

  X, y = np.array(X), np.array(y)

  split = int(len(X) * 0.8)
  xtrn, ytrn = X[:split], y[:split]
  xtst, ytst = X[split:], y[split:]

  xtrn = xtrn.reshape((xtrn.shape[0], xtrn.shape[1], 1))
  xtst = xtst.reshape((xtst.shape[0], xtst.shape[1], 1))

  model = models.Sequential([
      Input(shape=(xtrn.shape[1], 1)),
      LSTM(units=50, return_sequences=True),
      Dropout(0.2),
      LSTM(units=50, return_sequences=False),
      Dropout(0.2),
      Dense(units=25, activation="relu"),
      Dense(units=1),
  ])

  model.compile(optimizer="adam", loss="mse", metrics=["mae"])

  early_stop = callbacks.EarlyStopping(
      monitor="val_loss", patience=5, restore_best_weights=True
  )

  with st.spinner(
      f"Training custom neural network for {ticker}... Please wait."
  ):
    model.fit(
        xtrn,
        ytrn,
        validation_data=(xtst, ytst),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

  st.success("Training complete!")

  y_pred_scaled = model.predict(xtst)
  y_pred = scaler.inverse_transform(y_pred_scaled)
  ytst_actual = scaler.inverse_transform(ytst.reshape(-1, 1))

  mae = mean_absolute_error(ytst_actual, y_pred)
  rmse = np.sqrt(mean_squared_error(ytst_actual, y_pred))
  r2 = r2_score(ytst_actual, y_pred)

  actual_dir = np.sign(np.diff(ytst_actual.flatten()))
  pred_dir = np.sign(np.diff(y_pred.flatten()))
  dir_accuracy = np.mean(actual_dir == pred_dir) * 100

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("MAE", f"${mae:.2f}")
  col2.metric("RMSE", f"${rmse:.2f}")
  col3.metric("R² Score", f"{r2:.4f}")
  col4.metric("Directional Accuracy", f"{dir_accuracy:.2f}%")

  st.write("#### Test Set Evaluation: Actual vs Predicted")
  fig_eval = go.Figure()
  fig_eval.add_trace(
      go.Scatter(
          y=ytst_actual.flatten(),
          mode="lines",
          name="Actual Price",
          line=dict(color="#94A3B8", width=1.5),
      )
  )
  fig_eval.add_trace(
      go.Scatter(
          y=y_pred.flatten(),
          mode="lines",
          name="Predicted Price",
          line=dict(color="#EF4444", width=1.5, dash="dash"),
      )
  )
  fig_eval.update_layout(
      xaxis_title="Test Index",
      yaxis_title="Price (USD)",
      margin=dict(l=20, r=20, t=30, b=20),
      height=400,
  )
  st.plotly_chart(fig_eval, use_container_width=True)

  last_window = scaled_data[-lookback:].reshape(1, lookback, 1)
  next_scaled = model.predict(last_window)
  next_price = scaler.inverse_transform(next_scaled)[0][0]

  last_close = df["Close"].iloc[-1]
  change = next_price - last_close
  pct_change = (change / last_close) * 100

  st.write("#### Next Trading Session Prediction")
  st.metric(
      label=f"Projected Close for {ticker}",
      value=f"${next_price:.2f}",
      delta=f"${change:.2f} ({pct_change:.2f}%)",
  )