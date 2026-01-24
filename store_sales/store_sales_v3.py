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

##


train_all["promo_flag"] = (train_all["onpromotion"] > 0).astype(int)

train_all["promo_ratio"] = (
    train_all["onpromotion"] /
    (train_all.groupby("family")["onpromotion"].transform("mean") + 1e-6)
)


test_all["promo_flag"] = (test_all["onpromotion"] > 0).astype(int)

test_all["promo_ratio"] = (
    test_all["onpromotion"] /
    (test_all.groupby("family")["onpromotion"].transform("mean") + 1e-6)
)
##






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
num_features = ["onpromotion","promo_ratio","promo_flag","transactions","dcoilwtico","lag_1","lag_7","lag_14","lag_28" , "roll_mean_7","roll_mean_14","roll_mean_28","roll_std_7","roll_std_14","roll_std_28" ]





for col in cat_features:
    train_all[col] = train_all[col].astype("category")




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
            "city": grp[grp["store_nbr"]==store].iloc[0]["city"],
            "state":grp[grp["store_nbr"]==store].iloc[0]["state"],
            "type":grp[grp["store_nbr"]==store].iloc[0]["type"],
            "cluster":grp[grp["store_nbr"]==store].iloc[0]["cluster"],
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
        row["year"] = d.year

        row["is_weekend"] = int(d.weekday() >= 5)



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

        txn_val = history.loc[
        (history["date"] == d) &
        (history["store_nbr"] == store) &
        (history["family"] == family),
        "transactions"
        ]

        row["transactions"] = txn_val.iloc[0] if len(txn_val) > 0 else 0

        oil_val = history.loc[
        history["date"] == d,
        "dcoilwtico"
        ]

        row["dcoilwtico"] = oil_val.iloc[0] if len(oil_val) > 0 else np.nan


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


    for col in cat_features:
        df_step[col] = df_step[col].astype("category")

    return df_step


last_train_date = pd.to_datetime("2017-08-15")



forecast_dates = pd.date_range(
    start=last_train_date + pd.Timedelta(days=1),
    periods=16,
    freq="D"
)

history = train_all.copy()
predictions = []

for d in forecast_dates:
    X_step = create_features_from_history(history, d)
    # print(X_train.columns)
    # print(X_step.columns)

    y_pred = model.predict(X_step.drop("date",axis=1)  )
    

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
submission[["id","sales"]].to_csv("result.csv",index=False)