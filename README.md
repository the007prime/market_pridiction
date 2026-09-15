# 📈 StockPulse: Universal Stock Market Predictor & Dashboard

> An interactive web application built with **Streamlit**, **TensorFlow**, and **yfinance** that dynamically pulls real-time financial market data and forecasts upcoming close prices using a deep-learning Long Short-Term Memory (LSTM) neural network.

---

## ✨ Key Features

- **Universal Asset Search**: Query historical and live market data for any publicly listed stock ticker supported by Yahoo Finance (`AAPL`, `GOOG`, `TSLA`, `MSFT`, `RELIANCE.NS`, etc.).
- **Interactive Technical Charting**: Toggle between professional Plotly Candlestick charts and Simple Moving Averages (`MA100`, `MA200`).
- **On-the-Fly Deep Learning**: Dynamically scales time-series data and fits a custom stacked LSTM model tailored specifically to the asset of your choice.
- **Robust Model Evaluation**: Instant calculation of out-of-sample performance metrics including **MAE**, **RMSE**, **R² Score**, and **Directional Trend Accuracy**.
- **Next-Day Forecasting**: Automatically extracts trailing window records to project the immediate upcoming trading session's close price.

---

## 🛠️ Tech Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [TensorFlow / Keras](https://www.tensorflow.org/) (LSTM Architecture)
- **Data Manipulation & Finance**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [yfinance](https://github.com/ranaroussi/yfinance)
- **Data Visualization**: [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/)
- **Metrics & Scaling**: [Scikit-Learn](https://scikit-learn.org/)

---
