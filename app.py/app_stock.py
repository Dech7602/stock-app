import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# App Title
st.title("📈 G" \
"Google Stock Price Forecast with ARIMA")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("GOOG.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df = df.asfreq("B")  # Set Business day frequency
    return df[['Adj Close']]

data = load_data()

# Show raw data
st.subheader("📊 Historical Adjusted Close Price")
st.line_chart(data)

# Forecast horizon selector
forecast_days = st.slider("Select forecast horizon (days):", min_value=10, max_value=120, value=60)

# Build and fit ARIMA model
st.subheader(f"🔮 Forecast for next {forecast_days} business days")

try:
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

    # Generate forecast dates
    forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="B")
    forecast_series = pd.Series(forecast.values, index=forecast_dates)

    # Plot forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data[-200:], label="Historical (last 200 days)")
    ax.plot(forecast_series, label="Forecast", linestyle="--")
    ax.set_title("ARIMA Forecast of GOOG Stock Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Model failed: {e}")
