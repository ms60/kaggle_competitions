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

X= train.drop("Churn",axis=1)
y = train["Churn"]

X_test = test.drop("id",axis=1)

#------------------------------------------------


SUBSET_SIZE = 50_000

train_stratified, _ = train_test_split(
    train,
    train_size=SUBSET_SIZE,
    stratify=train["Churn"],
    random_state=42
)

X_stratified = train_stratified.drop("Churn", axis=1)
y_stratified = train_stratified["Churn"]


#------------------------------------------------

xgb_params = {
    'n_estimators': 60000,      
    'learning_rate': 0.01,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree':0.8,
    'max_bin':16000,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    #'early_stopping_rounds': 200,
    'device': 'cuda',
    
    'enable_categorical': True,
}

lgbm_params = {
    'n_estimators': 60000,
    'learning_rate': 0.002,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree':0.8,
    'max_bin':12000,
    'objective': 'binary',
    'metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    #'early_stopping_rounds': 200,
    #'device': 'cuda',
    'verbosity':-1,
}

estimators = [60000 , 70000 , 80000 , 90000 , 100000]
learning_rates = [0.006 , 0.005 , 0.004 , 0.003 , 0.002 ]
max_bins = [12000 ,16000 , 20000 , 24000 , 28000]
scores = []

X_train, X_valid, y_train, y_valid = train_test_split(X_stratified, y_stratified, test_size=0.1, random_state=42, stratify=y_stratified)


# for idx,est in enumerate(estimators):
#     print( f"{idx+1}/{len(estimators)}" )

#     # xgb_params.update({
#     #     'n_estimators': est,
#     #     #'learning_rate': learning_rates[idx],

#     # })
#     # model = XGBClassifier(**xgb_params)
#     # model.fit(X_stratified,y_stratified,
#     #         eval_set=[(X_stratified, y_stratified)],
#     #         verbose=0)
    
#     lgbm_params.update({
#         #'n_estimators': est,
#         'learning_rate': learning_rates[idx],
#         #'max_bin': max_bins[idx],
#     })
#     model = LGBMClassifier(**lgbm_params)
#     model.fit(X_train,y_train)
    
    
#     y_pred = model.predict_proba(X_valid)[:, 1]
#     score = roc_auc_score(y_valid, y_pred)
#     scores.append(score)

# import matplotlib.pyplot as plt

# plt.plot(scores)
# plt.show()


#-------------------------

#stats encoding
for col in ["PaymentMethod","Contract"]:
    stats = train.groupby(col)["Churn"].agg(["mean","std","count"])

    for s in stats.columns:
        X[f"{col}_{s}"] = X[col].map(stats[s]).astype("float32")
        X_test[f"{col}_{s}"] = X_test[col].map(stats[s]).astype("float32")
        

# rank encoding
for col in ["PaymentMethod","Contract"]:

    counts = X[col].value_counts()
    rank_map = counts.rank(method="dense", ascending=False)

    X[f"{col}_rank"] = X[col].map(rank_map).astype("int32")
    X_test[f"{col}_rank"] = X_test[col].map(rank_map).astype("int32")




import cupy as cp

for col in ["PaymentMethod","Contract"]:
    X[col] = X[col].astype("category").cat.codes
    X_test[col] = X_test[col].astype("category").cat.codes



X_gpu = cp.asarray(X.values.astype("float32"))
X_test_gpu = cp.asarray(X_test.values.astype("float32"))

model = XGBClassifier(**xgb_params)

model.fit(
    X_gpu,
    y,
    eval_set=[(X_gpu, y)],
    verbose=3000
)

booster = model.get_booster()

y_pred_gpu = booster.inplace_predict(X_test_gpu)
y_pred = cp.asnumpy(y_pred_gpu)

result = pd.DataFrame({"id": test["id"], "Churn": y_pred})
result.to_csv("./result_stratified.csv", index=False)