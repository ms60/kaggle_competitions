import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
#from lightgbm import early_stopping_ro


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
#store_nbr identifies the store at which the products are sold.
#family identifies the type of product sold.
#onpromotion gives the total number of items in a product family that were being promoted at a store at a given date.

train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])


# print(train.tail())

# print(test.head())
# print(train.shape)
# print(train.isnull().sum())

# print(train["store_nbr"].unique())


# Naive	yₜ₊₁ = yₜ
# Seasonal naive	yₜ₊ₛ
# Moving average	Local mean
# Drift	Linear extrapolation
# 📌 Baseline geçilemiyorsa → proje çöktü

#print(train.dtypes)

# print(train.groupby("store_nbr").sum(["sales"]).drop("id",axis=1))

# print(train.groupby("store_nbr").last("sales").drop("id",axis=1))

stores_train = []

for store in train["store_nbr"].unique():
    stores_train.append( train[train["store_nbr"]==store] )

#print(stores_train[0].head())

stores_test = []

for store in test["store_nbr"].unique():
    stores_test.append(test[ test["store_nbr"]==store ])


###################################################################
def naive_forecast(y, h=1):
    last_value = y.iloc[-1]
    return np.repeat(last_value, h)

#naive_forecast(y, h=5)

def seasonal_naive_forecast(y, s, h=1):
    return [y.iloc[-s + i % s] for i in range(h)]

# örnek: aylık veri, yıllık sezon
#seasonal_naive_forecast(y, s=12, h=6)

def moving_average_forecast(y, window, h=1):
    #Son k gözlemin ortalaması
    mean_value = y.iloc[-window:].mean()
    return np.repeat(mean_value, h)

#moving_average_forecast(y, window=5, h=5)


def rolling_ma_forecast(y, window, h):
    y_ext = y.copy()
    forecasts = []

    for _ in range(h):
        pred = y_ext.iloc[-window:].mean()
        forecasts.append(pred)
        y_ext = pd.concat([y_ext, pd.Series([pred])])

    return forecasts

#rolling_ma_forecast(y, window=5, h=5)

def drift_forecast(y, h=1):
    #İlk ve son noktaya bak
    #Lineer trend varsay
    T = len(y)
    drift = (y.iloc[-1] - y.iloc[0]) / (T - 1)
    return [y.iloc[-1] + drift * i for i in range(1, h+1)]

#drift_forecast(y, h=5)
###################################################################

# fig, ax = plt.subplots(6, 9, figsize=(18, 12) , sharex=True ,sharey=True)
# j = 0
# for i, store in enumerate(stores_train):
#     if i % 6 == 0 and i != 0:
#         j += 1
#     ax[i % 6, j].plot(store["date"], store["sales"])
#     ax[i % 6, j].set_title(f"store: {i+1}")


#print(test["date"].min() , test["date"].max())

#print(train["family"].unique())
#print(train["family"].nunique())

# plt.tight_layout()
# plt.show()

# plt.show()


##############################

train["dow"] = train["date"].dt.weekday          # 0-6
train["week"] = train["date"].dt.isocalendar().week.astype(int)
train["month"] = train["date"].dt.month
train["year"] = train["date"].dt.year
train["is_weekend"] = (train["dow"] >= 5).astype(int)

train["sin_dow"] = np.sin(2 * np.pi * train["dow"] / 7)
train["cos_dow"] = np.cos(2 * np.pi * train["dow"] / 7)

LAGS = [1, 7, 14, 28]

for lag in LAGS:
    train[f"lag_{lag}"] = (
        train.groupby(["store_nbr", "family"])["sales"]
          .shift(lag)
    )

WINDOWS = [7, 14, 28]

for w in WINDOWS:
    train[f"roll_mean_{w}"] = (
        train.groupby(["store_nbr", "family"])["sales"]
          .shift(1)
          .rolling(w)
          .mean()
    )

    train[f"roll_std_{w}"] = (
        train.groupby(["store_nbr", "family"])["sales"]
          .shift(1)
          .rolling(w)
          .std()
    )


train["promo_flag"] = (train["onpromotion"] > 0).astype(int)

train["promo_ratio"] = (
    train["onpromotion"] /
    (train.groupby("family")["onpromotion"].transform("mean") + 1e-6)
)

cat_features = ["store_nbr", "family"]

train_model = train#.dropna().reset_index(drop=True)

train_model["y"] = np.log1p(train_model["sales"])
TARGET = "y"

train_model["store_nbr"] = train_model["store_nbr"].astype("category")
train_model["family"]    = train_model["family"].astype("category")

#print(train_model.head())

#MODEL VALIDATION

cutoff_date = "2017-01-01"

train_df = train_model[train_model["date"] < cutoff_date]
test_df   = train_model[train_model["date"] >= cutoff_date]

#################




FEATURES = [
    "lag_1","lag_7","lag_14","lag_28",
    "roll_mean_7","roll_mean_14","roll_mean_28",
    "roll_std_7","roll_std_14","roll_std_28",
    "dow","week","month","is_weekend",
    "sin_dow","cos_dow",
    "onpromotion","promo_flag","promo_ratio",
    "store_nbr","family"
]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

#print(train.head())


#model


params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "seed": 42
}

train_set = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=cat_features,
    free_raw_data=False
)

test_set = lgb.Dataset(
    X_test,
    label=y_test,
    categorical_feature=cat_features,
    free_raw_data=False
)

model = lgb.train(
    params,
    train_set,
    valid_sets=[train_set, test_set],
    valid_names=["train", "test"],
    num_boost_round=2000,
    #early_stopping_rounds=100,
    #verbose_eval=100
)

val_pred_log = model.predict(X_test)
val_pred = np.expm1(val_pred_log)
y_true = np.expm1(y_test)

wape = np.sum(np.abs(y_true - val_pred)) / np.sum(y_true)
print("WAPE:", wape)


#################
# predict test



LAGS = [1, 7, 14, 28]
WINDOWS = [7, 14, 28]

def create_features_from_history(history, d):
    rows = []

    # sadece geçmişi al
    hist = history[history["date"] < d]

    for (store, family), grp in hist.groupby(["store_nbr", "family"]):
        grp = grp.sort_values("date")

        row = {
            "date": d,
            "store_nbr": store,
            "family": family,
        }

        # --- lag features ---
        for lag in LAGS:
            row[f"lag_{lag}"] = grp["sales"].iloc[-lag] if len(grp) >= lag else np.nan

        # --- rolling features ---
        for w in WINDOWS:
            if len(grp) >= w:
                row[f"roll_mean_{w}"] = grp["sales"].iloc[-w:].mean()
                row[f"roll_std_{w}"]  = grp["sales"].iloc[-w:].std()
            else:
                row[f"roll_mean_{w}"] = np.nan
                row[f"roll_std_{w}"]  = np.nan

        # --- calendar ---
        row["dow"] = d.weekday()
        row["week"] = d.isocalendar().week
        row["month"] = d.month
        row["is_weekend"] = int(d.weekday() >= 5)

        row["sin_dow"] = np.sin(2 * np.pi * row["dow"] / 7)
        row["cos_dow"] = np.cos(2 * np.pi * row["dow"] / 7)

        # --- promotion ---
        # ⚠️ varsayım: future promo biliniyor
        promo_val = history.loc[
            (history["date"] == d) &
            (history["store_nbr"] == store) &
            (history["family"] == family),
            "onpromotion"
        ]

        row["onpromotion"] = promo_val.iloc[0] if len(promo_val) > 0 else 0
        row["promo_flag"] = int(row["onpromotion"] > 0)

        rows.append(row)

    df_step = pd.DataFrame(rows)

    # promo_ratio (train ile aynı tanım!)
    family_mean_promo = (
        history.groupby("family")["onpromotion"].mean()
    )

    df_step["promo_ratio"] = (
        df_step["onpromotion"] /
        (df_step["family"].map(family_mean_promo) + 1e-6)
    )

    # categorical dtype
    df_step["store_nbr"] = df_step["store_nbr"].astype("category")
    df_step["family"] = df_step["family"].astype("category")

    return df_step


last_train_date = pd.to_datetime("2017-08-15")



forecast_dates = pd.date_range(
    start=last_train_date + pd.Timedelta(days=1),
    periods=16,
    freq="D"
)

history = train_df.copy()
predictions = []

for d in forecast_dates:
    X_step = create_features_from_history(history, d)

    y_pred_log = model.predict(X_step[FEATURES])
    y_pred = np.expm1(y_pred_log)

    X_step["sales"] = y_pred
    history = pd.concat([history, X_step], ignore_index=True)

    predictions.append(X_step)


# preds = []
# for pred_df in preds:

# print(predictions[0])
# print(predictions[-1])

pred_df = pd.concat(predictions, ignore_index=True)

submission = test.merge(
    pred_df[['date','store_nbr','family','sales']],
    on=['date','store_nbr','family'],
    how='left'
)

#submission["sales"] = np.expm1(submission["sales"])
submission = submission.sort_values("id")
submission.to_csv("result.csv",index=False)