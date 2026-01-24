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


print(test_all.iloc[[0],:])
print("-"*40)
first_date = test_all.iloc[[0],:]["date"]
first_store_nbr = test_all.iloc[[0],:]["store_nbr"][0]
first_family  = test_all.iloc[[0],:]["family"][0]
first_dow  = test_all.iloc[[0],:]["dow"][0]
first_week  = test_all.iloc[[0],:]["week"][0]
first_month  = test_all.iloc[[0],:]["month"][0]


first_df = train_all.groupby(["store_nbr","family"]).get_group((first_store_nbr, first_family ) )
first_df_dow = first_df.groupby(["dow"]).get_group(first_dow)
first_df_dow_w = first_df.groupby(["dow","week"]).get_group((first_dow,first_week))
first_df_w_m = first_df.groupby(["week","month"]).get_group((first_week,first_month))
first_df_dow_w_m = first_df.groupby(["dow","week","month"]).get_group((first_dow,first_week,first_month))


print( first_df["sales"].mean() )
print( first_df_dow["sales"].mean() )
print( first_df_dow_w["sales"].mean() )
print( first_df_w_m["sales"].mean() )
print( first_df_dow_w_m["sales"].mean() )


# full_range = pd.date_range(
#     start=train_all["date"].min(),
#     end=train_all["date"].max(),
#     freq="D"
# )

# missing_dates = full_range.difference(train_all["date"])
# print(len(missing_dates))


# print(first_store_nbr,first_family)
# print( first_df)



# create_lag_features(train_all , LAGS)
# create_rolling_features(train_all,WINDOWS)

# LAGS = [1,  3 , 7, 10]
# WINDOWS = [3, 7, 10]

# baseline_ids = []
# baseline_predicts = []

# for i,d in enumerate( test_all["date"].to_list(),start=0):
#     the_date = test_all.iloc[[i],:]["date"][i]
#     the_store_nbr = test_all.iloc[[i],:]["store_nbr"][i]
#     the_family  = test_all.iloc[[i],:]["family"][i]
#     the_dow  = test_all.iloc[[i],:]["dow"][i]
#     the_week  = test_all.iloc[[i],:]["week"][i]
#     the_month  = test_all.iloc[[i],:]["month"][i]


    

#     the_df = train_all.groupby(["store_nbr","family"]).get_group((the_store_nbr, the_family ) )
#     the_df_dow = the_df.groupby(["dow"]).get_group(the_dow)
#     the_df_dow_w = the_df.groupby(["dow","week"]).get_group((the_dow,the_week))
#     the_df_w_m = the_df.groupby(["week","month"]).get_group((the_week,the_month))
#     the_df_dow_w_m = the_df.groupby(["dow","week","month"]).get_group((the_dow,the_week,the_month))

#     total = the_df["sales"].mean() + the_df_dow["sales"].mean() + the_df_dow_w["sales"].mean() + the_df_w_m["sales"].mean() + the_df_dow_w_m["sales"].mean()

#     baseline_ids.append( test_all.iloc[[i],:]["id"][i] )
#     baseline_predicts.append( total / 5.0 )

# result = pd.DataFrame( {"id":baseline_ids , "sales":baseline_predicts} )
# result.to_csv("result.csv",index=False)


baseline_ids = []
baseline_predicts = []

g_sf = train_all.groupby(["store_nbr","family"])["sales"].mean()

g_dow = train_all.groupby(["store_nbr","family","dow"])["sales"].mean()

g_dow_w = train_all.groupby(
    ["store_nbr","family","dow","week"]
)["sales"].mean()

g_w_m = train_all.groupby(
    ["store_nbr","family","week","month"]
)["sales"].mean()

g_dow_w_m = train_all.groupby(
    ["store_nbr","family","dow","week","month"]
)["sales"].mean()

def safe_get(series, key):
    try:
        return series.loc[key]
    except KeyError:
        return np.nan

for _, row in test_all.iterrows():
    key_sf = (row.store_nbr, row.family)
    key_dow = (row.store_nbr, row.family, row.dow)
    key_dow_w = (row.store_nbr, row.family, row.dow, row.week)
    key_w_m = (row.store_nbr, row.family, row.week, row.month)
    key_dow_w_m = (row.store_nbr, row.family, row.dow, row.week, row.month)

    values = [
        safe_get(g_sf, key_sf),
        safe_get(g_dow, key_dow),
        safe_get(g_dow_w, key_dow_w),
        safe_get(g_w_m, key_w_m),
        safe_get(g_dow_w_m, key_dow_w_m),
    ]

    pred = np.nanmean(values)  # varsa olanları alır

    baseline_ids.append(row.id)
    baseline_predicts.append(pred)

result = pd.DataFrame({
    "id": baseline_ids,
    "sales": baseline_predicts
})

result.to_csv("result.csv", index=False)