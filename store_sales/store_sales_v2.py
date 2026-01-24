import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
#from lightgbm import early_stopping_ro
from sklearn.preprocessing import MinMaxScaler , StandardScaler , OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.metrics import mean_squared_error


from datetime import date, timedelta
import datetime

#pd.set_option('display.max_columns', 500)

def daterange(start_date: date, end_date: date):
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)




oil = pd.read_csv("./data/oil.csv")
oil["date"] = pd.to_datetime( oil["date"] )




date_min = oil["date"].min()
date_max = oil["date"].max()

dateList = []
for single_date in daterange(date_min, date_max):
    dateList.append(single_date)

oil_new = pd.DataFrame( {"date":dateList} )


oil_new = pd.merge( oil_new , oil , how="left" , on="date" )
oil_new.iloc[0,1] = oil_new.iloc[1,1]

oil_new = oil_new.bfill()

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
#store_nbr identifies the store at which the products are sold.
#family identifies the type of product sold.
#onpromotion gives the total number of items in a product family that were being promoted at a store at a given date.

train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])





transactions = pd.read_csv("./data/transactions.csv")
transactions["date"] = pd.to_datetime(transactions["date"])

holiday_events = pd.read_csv("./data/holidays_events.csv")
holiday_events["date"] = pd.to_datetime(holiday_events["date"])


stores = pd.read_csv("./data/stores.csv")



train_transactions = pd.merge(  train , transactions , how="left", on=["date","store_nbr"] )
train_transactions["transactions"] = train_transactions["transactions"].fillna(0) 

train_transactions_stores = pd.merge(train_transactions , stores , how="left", on="store_nbr")

train_transactions_stores_oil = pd.merge(train_transactions_stores , oil_new , how="left",on = "date")


train_all = train_transactions_stores_oil.copy()

### test
test_transactions = pd.merge(  test , transactions , how="left", on=["date","store_nbr"] )
test_transactions["transactions"] = test_transactions["transactions"].fillna(0) 

test_transactions_stores = pd.merge(test_transactions , stores , how="left", on="store_nbr")

test_transactions_stores_oil = pd.merge(test_transactions_stores , oil_new , how="left",on = "date")



###

test_all = test_transactions_stores_oil.copy()








#####
# FEATURE ENGINEERING

train_all["dow"] = train_all["date"].dt.weekday
train_all["week"] = train_all["date"].dt.isocalendar().week.astype(int)
train_all["month"] = train_all["date"].dt.month
train_all["year"] = train_all["date"].dt.year
train_all["is_weekend"] = (train_all["dow"] >= 5).astype(int)

# train_all["sin_dow"] = np.sin(2 * np.pi * train_all["dow"] / 7)
# train_all["cos_dow"] = np.cos(2 * np.pi * train_all["dow"] / 7)

test_all["dow"] = test_all["date"].dt.weekday
test_all["week"] = test_all["date"].dt.isocalendar().week.astype(int)
test_all["month"] = test_all["date"].dt.month
test_all["year"] = test_all["date"].dt.year
test_all["is_weekend"] = (test_all["dow"] >= 5).astype(int)




def create_lag_features(df , lags):

    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(["store_nbr", "family"])["sales"]
            .shift(lag)
        )

LAGS = [1,  3 , 7, 10]
create_lag_features(train_all , LAGS)




def create_rolling_features(df,windows):

    for w in windows:
        df[f"roll_mean_{w}"] = (
            df.groupby(["store_nbr", "family"])["sales"]
            .shift(1)
            .rolling(w)
            .mean()
        )

        df[f"roll_std_{w}"] = (
            df.groupby(["store_nbr", "family"])["sales"]
            .shift(1)
            .rolling(w)
            .std()
        )

WINDOWS = [3, 7, 10]
create_rolling_features(train_all,WINDOWS)


print(train_all)

train_all = train_all.dropna()

cat_features = ["store_nbr", "family","city","state","type","cluster","dow"] # ,"week","month","year"
num_features = ["onpromotion","transactions","dcoilwtico","lag_1","lag_7","lag_14","lag_28" , "roll_mean_7","roll_mean_14","roll_mean_28","roll_std_7","roll_std_14","roll_std_28" ]


for col in cat_features:
    train_all[col] = train_all[col].astype("category")

for col in ["lag_1","lag_3","lag_7","lag_10",
            "roll_mean_3","roll_mean_7","roll_mean_10",
            "roll_std_3","roll_std_7","roll_std_10"]:
    test_all[col] = np.nan


for col in cat_features:
    test_all[col] = test_all[col].astype("category")


#Time series için tarih sırası önemli
cutoff_date = datetime.datetime.strptime("2017-07-01", "%Y-%m-%d") 

train_mask = train_all["date"] < cutoff_date
valid_mask = train_all["date"] >= cutoff_date

X_train = train_all.loc[train_mask].drop(["id","sales","date"], axis=1)
X_train_with_date = train_all.loc[train_mask].drop(["id","sales"], axis=1)
y_train = train_all.loc[train_mask]["sales"]

X_valid = train_all.loc[valid_mask].drop(["id","sales","date"], axis=1)
X_valid_with_date = train_all.loc[valid_mask].drop(["id","sales"], axis=1)
y_valid = train_all.loc[valid_mask]["sales"]




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

model = lgb.LGBMRegressor(**params)
model.fit(X_train , y_train , categorical_feature=cat_features)


#ilk günün tahmini , 

print(model.predict( X_valid.iloc[[0],:] ))
# bunu test zamanı boyunca , lag,rolling feature larını güncelleyerek recursive tahmin et



# all_preds = []

# for (store, family), df_group in X_valid_with_date.groupby(["store_nbr", "family"]):
#     # 1. tarih sıralama
#     df_group = df_group.sort_values("date").copy()

#     # 2. history oluştur (train dönemi)
#     history = train_all[
#         (train_all["store_nbr"] == store) &
#         (train_all["family"] == family)
#     ][["date", "sales"]].copy()

#     # 3. recursive loop
#     for t in df_group["date"]:
#         X_step = df_group[df_group["date"] == t].copy()

#         # lag & rolling feature güncelle (history'den)
#         X_step["lag_1"] = history["sales"].iloc[-1]
#         X_step["lag_3"] = history["sales"].iloc[-3]
#         X_step["lag_7"] = history["sales"].iloc[-7]
#         X_step["lag_10"] = history["sales"].iloc[-10]

#         X_step["roll_mean_3"] = history["sales"].iloc[-3:].mean()
#         X_step["roll_mean_7"] = history["sales"].iloc[-7:].mean()
#         X_step["roll_mean_10"] = history["sales"].iloc[-10:].mean()


#         X_step["roll_std_3"] = history["sales"].iloc[-3:].std()
#         X_step["roll_std_7"] = history["sales"].iloc[-7:].std()
#         X_step["roll_std_10"] = history["sales"].iloc[-10:].std()

#         # predict
#         y_pred = model.predict(X_step.drop("date",axis=1))[0]

#         # history update
#         history = pd.concat([history, pd.DataFrame([{"date": t, "sales": y_pred}])],
#                             ignore_index=True)

#         all_preds.append({
#             "store_nbr": store,
#             "family": family,
#             "date": t,
#             "pred": y_pred
#         })

# # sonunda DataFrame'e çevir
# preds_df = pd.DataFrame(all_preds)
# print(preds_df)

# X_valid_with_preds = X_valid_with_date.merge(
#     preds_df,
#     on=["store_nbr", "family", "date"],
#     how="left"
# )

# y_true = y_valid.values

# y_pred = X_valid_with_preds["pred"].values

# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# print("RMSE:", rmse)

# epsilon = 1e-5  # sıfırdan kaçınmak için küçük değer
# y_true_log = np.log(y_true + epsilon)
# y_pred_log = np.log(y_pred + epsilon)

# log_rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
# print("Log RMSE:", log_rmse)



all_preds_test = []

for (store, family), df_group in test_all.groupby(["store_nbr", "family"]):
    # Tarihe göre sırala
    df_group = df_group.sort_values("date").copy()

    # History = train döneminin son n günü
    history = train_all[
        (train_all["store_nbr"] == store) &
        (train_all["family"] == family)
    ][["date", "sales"]].copy()

    for t in df_group["date"]:
        X_step = df_group[df_group["date"] == t].copy()

        # Lag & rolling features (history’den)
        X_step["lag_1"]  = history["sales"].iloc[-1]
        X_step["lag_3"]  = history["sales"].iloc[-3]
        X_step["lag_7"]  = history["sales"].iloc[-7]
        X_step["lag_10"] = history["sales"].iloc[-10]

        X_step["roll_mean_3"]  = history["sales"].iloc[-3:].mean()
        X_step["roll_mean_7"]  = history["sales"].iloc[-7:].mean()
        X_step["roll_mean_10"] = history["sales"].iloc[-10:].mean()

        X_step["roll_std_3"]  = history["sales"].iloc[-3:].std()
        X_step["roll_std_7"]  = history["sales"].iloc[-7:].std()
        X_step["roll_std_10"] = history["sales"].iloc[-10:].std()



        # Predict
        y_pred = model.predict(X_step.drop(["id","date"], axis=1))[0]

        # History update (recursive)
        history = pd.concat(
            [history, pd.DataFrame([{"date": t, "sales": y_pred}])],
            ignore_index=True
        )

        # Tahminleri kaydet
        all_preds_test.append({
            "store_nbr": store,
            "family": family,
            "date": t,
            "pred": y_pred
        })

# DataFrame’e çevir
preds_test_df = pd.DataFrame(all_preds_test)


test_with_preds = test_all.merge(
    preds_test_df,
    on=["store_nbr", "family", "date"],
    how="left"
)
test_with_preds.to_csv("result.csv")