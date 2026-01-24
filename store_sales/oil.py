import pandas as pd
import numpy as np
from datetime import date, timedelta

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

holiday_events = pd.read_csv("./data/holidays_events.csv")
holiday_events["date"] = pd.to_datetime(holiday_events["date"])

print(holiday_events)

holiday_events_new = pd.DataFrame({"date":dateList})
holiday_events_new = pd.merge(holiday_events_new , holiday_events , how="left",on="date")
print(holiday_events_new)




# transactions = pd.read_csv("./data/transactions.csv")
# transactions["date"] = pd.to_datetime(transactions["date"])


# stores = pd.read_csv("./data/stores.csv")


# oil_transactions = pd.merge( transactions , oil , how="left" , on="date")


# oil_transactions_stores = pd.merge(oil_transactions , stores , how="left" , on="store_nbr")


# holiday_events = pd.read_csv("./data/holidays_events.csv")
# holiday_events["date"] = pd.to_datetime(holiday_events["date"])



# oil_transactions_stores_holiday = pd.merge( oil_transactions_stores , holiday_events , how = "left", on="date" )

# print(oil_transactions_stores_holiday)

# train = pd.read_csv("./data/train.csv")
# train["date"] = pd.to_datetime(train["date"])
# print(train)

# train_all = pd.merge(train , oil_transactions_stores_holiday ,  how="left" , on=["date","store_nbr"])
# print(train_all)