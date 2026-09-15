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
    # auto_adjust=True stabilizes downloads on cloud IPs
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
      return None

    # yfinance may return MultiIndex columns like ("Close", "GOOG").
    # Flatten robustly regardless of level names/order.
    if isinstance(df.columns, pd.MultiIndex):
      if "Ticker" in df.columns.names:
        try:
          df = df.xs(symbol, level="Ticker", axis=1)
        except KeyError:
          df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
      else:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Also flatten any stray tuple-named columns.
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()

    if "Close" not in df.columns:
      return None

    # Ensure Close is a 1-D numeric series (guards against duplicate cols).
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
      close = close.iloc[:, 0]
    df["Close"] = pd.to_numeric(close, errors="coerce")
    df = df.dropna(subset=["Close"])
    if df.empty:
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

if "results" not in st.session_state:
  st.session_state["results"] = {}

run_model = st.button(f"Run Prediction Model for {ticker}", type="primary")

if run_model:
  # --- 1. Prepare raw series as clean 1-D float vector ---
  close_raw = df["Close"]
  if isinstance(close_raw, pd.DataFrame):
    close_raw = close_raw.iloc[:, 0]
  close_raw = pd.to_numeric(close_raw, errors="coerce").dropna().values.astype(float).reshape(-1, 1)

  # --- 2. Split BEFORE scaling to avoid data leakage ---
  # Keep `lookback` overlap so test sequences have full history.
  split_idx = int(len(close_raw) * 0.8)
  train_raw = close_raw[:split_idx]
  test_raw_with_history = close_raw[split_idx - lookback :]

  scaler = MinMaxScaler(feature_range=(0, 1))
  train_scaled = scaler.fit_transform(train_raw)
  test_scaled = scaler.transform(test_raw_with_history)

  def make_sequences(data, window):
    Xs, ys = [], []
    for i in range(window, len(data)):
      Xs.append(data[i - window : i, 0])
      ys.append(data[i, 0])
    return np.array(Xs), np.array(ys)

  xtrn, ytrn = make_sequences(train_scaled, lookback)
  xtst, ytst = make_sequences(test_scaled, lookback)

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

  y_pred_scaled = model.predict(xtst, verbose=0)
  y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
  ytst_actual = scaler.inverse_transform(ytst.reshape(-1, 1)).flatten()

  mae = mean_absolute_error(ytst_actual, y_pred)
  rmse = float(np.sqrt(mean_squared_error(ytst_actual, y_pred)))
  r2 = r2_score(ytst_actual, y_pred) if len(ytst_actual) > 1 else float("nan")

  if len(ytst_actual) > 1:
    actual_dir = np.sign(np.diff(ytst_actual))
    pred_dir = np.sign(np.diff(y_pred))
    # Ignore flat (0-change) steps so 0 vs 0 doesn't inflate/deflate score.
    mask = (actual_dir != 0) & (pred_dir != 0)
    dir_accuracy = float(np.mean(actual_dir[mask] == pred_dir[mask]) * 100) if mask.any() else 0.0
  else:
    dir_accuracy = 0.0

  # Next-session forecast from the most recent (scaled with train-fit scaler) window.
  last_window_raw = close_raw[-lookback:].reshape(-1, 1)
  last_window = scaler.transform(last_window_raw).reshape(1, lookback, 1)
  next_scaled = model.predict(last_window, verbose=0)
  next_price = float(scaler.inverse_transform(next_scaled)[0][0])

  last_close = float(close_raw[-1][0])
  change = next_price - last_close
  pct_change = (change / last_close) * 100 if last_close != 0 else 0.0

  st.session_state["results"][ticker] = {
      "mae": float(mae),
      "rmse": float(rmse),
      "r2": float(r2),
      "dir_accuracy": float(dir_accuracy),
      "y_pred": y_pred.tolist(),
      "y_actual": ytst_actual.tolist(),
      "next_price": float(next_price),
      "last_close": float(last_close),
      "change": float(change),
      "pct_change": float(pct_change),
      "lookback": int(lookback),
      "epochs": int(epochs),
      "batch_size": int(batch_size),
  }
  st.success("Training complete!")

# --- 3. Render persisted results (survives Streamlit re-runs) ---
res = st.session_state["results"].get(ticker)
if res is not None:
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("MAE", f"${res['mae']:.2f}")
  col2.metric("RMSE", f"${res['rmse']:.2f}")
  col3.metric("R² Score", f"{res['r2']:.4f}")
  col4.metric("Directional Accuracy", f"{res['dir_accuracy']:.2f}%")

  st.write("#### Test Set Evaluation: Actual vs Predicted")
  fig_eval = go.Figure()
  fig_eval.add_trace(
      go.Scatter(
          y=res["y_actual"],
          mode="lines",
          name="Actual Price",
          line=dict(color="#94A3B8", width=1.5),
      )
  )
  fig_eval.add_trace(
      go.Scatter(
          y=res["y_pred"],
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

  st.write("#### Next Trading Session Prediction")
  st.metric(
      label=f"Projected Close for {ticker}",
      value=f"${res['next_price']:.2f}",
      delta=f"${res['change']:.2f} ({res['pct_change']:.2f}%)",
  )