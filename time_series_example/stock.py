import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA


np.random.seed(42)

## CREATE SYNTETIC DATA

# Zaman ekseni
n_days = 3 * 365  # 3 yıl
dates = pd.date_range(start="2021-01-01", periods=n_days, freq="D")

# Trend (yavaş yukarı giden)
trend = 0.02 * np.arange(n_days)

# Seasonality (yıllık + haftalık)
yearly_seasonality = 5 * np.sin(2 * np.pi * np.arange(n_days) / 365)
weekly_seasonality = 2 * np.sin(2 * np.pi * np.arange(n_days) / 7)

# Noise
noise = np.random.normal(0, 1.5, n_days)

# Stock price
price = 50 + trend + yearly_seasonality + weekly_seasonality + noise

# DataFrame
df = pd.DataFrame({
    "date": dates,
    "stock": price
}).set_index("date")

print(df.head())

# plt.figure(figsize=(14, 5))
# plt.plot(df["stock"])
# plt.show()

# differencing transform yt = yt-1

diff = df["stock"].diff()

# plt.plot(diff)
# plt.show()


# log transform yt=log(yt)

log = np.log(df["stock"])
# plt.plot(log)
# plt.show()

# log + diff

log_diff = log.diff().dropna()
# plt.plot(log_diff)
# plt.show()

print(log_diff.head())

#stationary test Augmented Dickey-Fuller test

stationary_test = adfuller(log_diff)
print("ADF statistics:",stationary_test[0])
print("P-Value:",stationary_test[1])



# box-cox transformation to make distrubtion into normal-dist.
#1 variance stabilization
data_box_cox ,lam = boxcox(df["stock"])

# plt.plot(data_box_cox)
# plt.text(0.5, 0.5, lam, dict(size=10))
# plt.show()
print(lam)
print(data_box_cox)

# mean stabilization
data_box_cox_diff = np.diff(data_box_cox)

# remove seasonality 
s = 365
data_box_cox_diff_seasonal = data_box_cox_diff[s:] - data_box_cox_diff[:-s]
print(pd.Series(data_box_cox_diff_seasonal).describe())

acf_vals = acf(data_box_cox_diff_seasonal, nlags=10)
pacf_vals = pacf(data_box_cox_diff_seasonal, nlags=10)

print(acf_vals, pacf_vals)

n = len(data_box_cox_diff_seasonal)
train_size = int(n * 0.8)

train = data_box_cox_diff_seasonal[:train_size]
test  = data_box_cox_diff_seasonal[train_size:]

model = ARIMA(train, order=(1, 0, 0))  # AR(1)
result = model.fit()

n_forecast = len(test)

forecast_diff_seasonal = result.forecast(steps=n_forecast)


last_season = data_box_cox_diff[-365:]

reconstructed_diff = []

for i, val in enumerate(forecast_diff_seasonal):
    base = last_season[i] if i < 365 else reconstructed_diff[i-365]
    reconstructed_diff.append(val + base)

reconstructed_diff = np.array(reconstructed_diff)

last_value = data_box_cox[-1]

y_bc_reconstructed = last_value + np.cumsum(reconstructed_diff)

from scipy.special import inv_boxcox

y_pred = inv_boxcox(y_bc_reconstructed, lam)

y = df["stock"].values
true_prices = y[-len(y_pred):]

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(true_prices, y_pred)
print(mae)

actual = y[-len(y_pred):]

# index oluşturalım (günlük)
dates = pd.date_range(
    start="2021-01-01",
    periods=len(y),
    freq="D"
)

actual_index = dates[-len(y_pred):]

plt.figure(figsize=(14,6))

plt.plot(actual_index, actual, label="Actual", linewidth=2)
plt.plot(actual_index, y_pred, label="Forecast (ARIMA)", linestyle="--")

plt.title("Actual vs Forecast (Inverse Transformed)")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(alpha=0.3)

plt.show()