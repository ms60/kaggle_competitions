from itertools import combinations, product
import random
from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from matplotlib import pyplot as plt
import seaborn as sns

import shap
from tqdm import tqdm
from xgboost import XGBClassifier


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

train["gender"] = train["gender"].map({"Female": 0, "Male": 1})
test["gender"] = test["gender"].map({"Female": 0, "Male": 1})

train["Partner"] = train["Partner"].map({"No": 0, "Yes": 1})
test["Partner"] = test["Partner"].map({"No": 0, "Yes": 1})

train["Dependents"] = train["Dependents"].map({"No": 0, "Yes": 1})
test["Dependents"] = test["Dependents"].map({"No": 0, "Yes": 1})

train["PhoneService"] = train["PhoneService"].map({"No": 0, "Yes": 1})
test["PhoneService"] = test["PhoneService"].map({"No": 0, "Yes": 1})

train["MultipleLines"] = train["MultipleLines"].map({"No": 0, "Yes": 1, "No phone service": 0})
test["MultipleLines"] = test["MultipleLines"].map({"No": 0, "Yes": 1, "No phone service": 0})

train["InternetService"] = train["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2})
test["InternetService"] = test["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2})

train["OnlineSecurity"] = train["OnlineSecurity"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["OnlineSecurity"] = test["OnlineSecurity"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["OnlineBackup"] = train["OnlineBackup"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["OnlineBackup"] = test["OnlineBackup"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["DeviceProtection"] = train["DeviceProtection"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["DeviceProtection"] = test["DeviceProtection"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["TechSupport"] = train["TechSupport"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["TechSupport"] = test["TechSupport"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["StreamingTV"] = train["StreamingTV"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["StreamingTV"] = test["StreamingTV"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["StreamingMovies"] = train["StreamingMovies"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["StreamingMovies"] = test["StreamingMovies"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["PaperlessBilling"] = train["PaperlessBilling"].map({"No": 0, "Yes": 1})
test["PaperlessBilling"] = test["PaperlessBilling"].map({"No": 0, "Yes": 1})

train["Churn"] = train["Churn"].map({"No": 0, "Yes": 1})

train = train.drop("id",axis=1)

for col in ["PaymentMethod","Contract"]:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")



#--------------------------------------------------

X = train.drop("Churn",axis=1)
y = train["Churn"]

X_test = test.drop("id",axis=1)

numeric_cols = ["tenure","MonthlyCharges","TotalCharges"]

#------------------------------------------------




# X_numeric = X.loc[0:100_000,numeric_cols]

# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler

# X_scaled = StandardScaler().fit_transform(X_numeric)

# nn = NearestNeighbors(
#     n_neighbors=5,   # her nokta için en yakın 5 komşu
#     metric="cosine",
#     algorithm="auto"
# )

# nn.fit(X_scaled)

# distances, indices = nn.kneighbors(X_scaled)

# duplicates = set()

# threshold = 0.001  # cosine distance (küçük = benzer)

# for i in range(len(indices)):
#     for j_idx, dist in zip(indices[i], distances[i]):
#         if i != j_idx and dist < threshold:
#             duplicates.add(j_idx)

# print(duplicates)
#----

# df_round = X.copy()
# df_round[numeric_cols] = df_round[numeric_cols].round(1)

# df_clean = df_round.drop_duplicates()

# print(df_round.shape)
# print(df_clean.shape)

#----
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

X_numeric = X[numeric_cols]
X_scaled = StandardScaler().fit_transform(X_numeric)

clustering = DBSCAN(
    eps=0.05,        # önemli hyperparam
    min_samples=2,
    metric="cosine"
).fit(X_scaled)

X["cluster"] = clustering.labels_
df_clean = X.groupby("cluster").first()

print(df_clean.shape)