import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

n = 3 * 365  # 3 yıl günlük
t = np.arange(n)

trend = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 365)
noise = np.random.normal(0, 3, n)

price = 100 + trend + seasonal + noise
price = np.exp(price / 100) * 50  # variance büyüsün diye

df = pd.DataFrame({"stock": price})

from scipy.stats import boxcox

y = df["stock"].values
y_bc, lam = boxcox(y)

y_bc_diff = np.diff(y_bc)

s = 365
y_bc_diff_seasonal = y_bc_diff[s:] - y_bc_diff[:-s]

from statsmodels.tsa.stattools import acf, pacf

acf_vals = acf(y_bc_diff_seasonal, nlags=10)
pacf_vals = pacf(y_bc_diff_seasonal, nlags=10)

n = len(y_bc_diff_seasonal)
train_size = int(n * 0.8)

train = y_bc_diff_seasonal[:train_size]
test  = y_bc_diff_seasonal[train_size:]

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=(1, 0, 0))  # AR(1)
result = model.fit()

print(result.summary())

n_forecast = len(test)

forecast_diff_seasonal = result.forecast(steps=n_forecast)


# seasonal anchor
last_season = y_bc_diff[-365:]

reconstructed_diff = []

for i, val in enumerate(forecast_diff_seasonal):
    base = last_season[i] if i < 365 else reconstructed_diff[i-365]
    reconstructed_diff.append(val + base)

reconstructed_diff = np.array(reconstructed_diff)

last_value = y_bc[-1]

y_bc_reconstructed = last_value + np.cumsum(reconstructed_diff)

from scipy.special import inv_boxcox

y_pred = inv_boxcox(y_bc_reconstructed, lam)

true_prices = y[-len(y_pred):]

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(true_prices, y_pred)
print("MAE:",mae)

# actual değerler (test period)
actual = y[-len(y_pred):]

# index oluşturalım (günlük)
dates = pd.date_range(
    start="2020-01-01",
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


h = len(y_pred)

last_train_value = y[-h-1]  # train'in son gerçek değeri
naive_pred = np.repeat(last_train_value, h)

s = 365

seasonal_naive_pred = y[-h-s:-s]

plt.figure(figsize=(15,6))

plt.plot(actual_index, actual, label="Actual", linewidth=2)

plt.plot(actual_index, y_pred,
         label="ARIMA Forecast", linestyle="--")

plt.plot(actual_index, naive_pred,
         label="Naive Baseline", linestyle=":")

plt.plot(actual_index, seasonal_naive_pred,
         label="Seasonal Naive Baseline", linestyle="-.")

plt.title("Actual vs Forecasts (ARIMA vs Baselines)")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(alpha=0.3)

plt.show()


from sklearn.metrics import mean_absolute_error

mae_arima = mean_absolute_error(actual, y_pred)
mae_naive = mean_absolute_error(actual, naive_pred)
mae_seasonal = mean_absolute_error(actual, seasonal_naive_pred)

print(mae_arima, mae_naive, mae_seasonal)

# 64.81312173411419 9.477287819832723 36.082441070987976