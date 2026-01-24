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




print(train_all)



cat_features = ["store_nbr", "family","city","state","type","cluster","dow"] # ,"week","month","year"
num_features = ["onpromotion","transactions","dcoilwtico","lag_1","lag_7","lag_14","lag_28" , "roll_mean_7","roll_mean_14","roll_mean_28","roll_std_7","roll_std_14","roll_std_28" ]


for col in cat_features:
    train_all[col] = train_all[col].astype("category")

print(train_all)
print("-"*40)
print(test_all)
print("-"*40)


features = test_all.drop( ["id"],axis=1 ).columns.tolist()

print( test_all.drop("id",axis=1).groupby([ "store_nbr","family" ]).count() ) 
