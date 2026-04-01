from lightgbm import LGBMClassifier , early_stopping
import optuna
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler , TargetEncoder , LabelEncoder

from sklearn.feature_selection import mutual_info_classif
import shap


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

train["Is_Long_Tenure"] = (train["tenure"] > 24).astype(int)
test["Is_Long_Tenure"] = (test["tenure"] > 24).astype(int)




# train["InternetQuality"] =   train["OnlineSecurity"] + train["OnlineBackup"] + train["DeviceProtection"] + train["TechSupport"] + train["StreamingTV"] + train["StreamingMovies"]
# test["InternetQuality"] =  test["OnlineSecurity"] + test["OnlineBackup"] + test["DeviceProtection"] + test["TechSupport"] + test["StreamingTV"] + test["StreamingMovies"] 

# train["Security"] = train["OnlineSecurity"] * train["DeviceProtection"] * train["TechSupport"]
# test["Security"] = test["OnlineSecurity"] * test["DeviceProtection"] * test["TechSupport"]

train['Expected_TotalCharges'] = train['tenure'] * train['MonthlyCharges']
test['Expected_TotalCharges'] = test['tenure'] * test['MonthlyCharges']

# train['Charges_Difference'] = train['TotalCharges'] - train['Expected_TotalCharges']
# test['Charges_Difference'] = test['TotalCharges'] - test['Expected_TotalCharges']

train['Cost_Per_Month_Tenure'] = train['TotalCharges'] / (train['tenure'] + 1)
test['Cost_Per_Month_Tenure'] = test['TotalCharges'] / (test['tenure'] + 1)

services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
train['Total_Services'] = train[services].sum(axis=1)
test['Total_Services'] = test[services].sum(axis=1)


train['Avg_Cost_Per_Service'] = train['MonthlyCharges'] / (train['Total_Services'] + 1)
test['Avg_Cost_Per_Service'] = test['MonthlyCharges'] / (test['Total_Services'] + 1)



contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
train['Contract_Months'] = train['Contract'].map(contract_map).astype(float).fillna(1.0)
test['Contract_Months'] = test['Contract'].map(contract_map).astype(float).fillna(1.0)


train['Tenure_Contract_Ratio'] = train['tenure'] / train['Contract_Months']
test['Tenure_Contract_Ratio'] = test['tenure'] / test['Contract_Months']

# train["Is_Safe_Customer"] = train["OnlineSecurity"] * train["TechSupport"]
# test["Is_Safe_Customer"] = test["OnlineSecurity"] * test["TechSupport"]


train['Is_Auto_Payment'] = train['PaymentMethod'].astype(str).str.contains('automatic', case=False).astype(int)
test['Is_Auto_Payment'] = test['PaymentMethod'].astype(str).str.contains('automatic', case=False).astype(int)

# train["Is_Streaming_Fan"] = train["StreamingMovies"] * train["StreamingTV"]
# test["Is_Streaming_Fan"] = test["StreamingMovies"] * test["StreamingTV"]

# train['Monthly_per_Contract_Month'] = train['MonthlyCharges'] / train['Contract_Months']
# test['Monthly_per_Contract_Month'] = test['MonthlyCharges'] / test['Contract_Months']


# train["tenure_treshold"] = (train["tenure"] >40.0).astype(int)
# test["tenure_treshold"] = (test["tenure"] >40.0).astype(int)

# train["tenure_squared"] = train["tenure"] * train["tenure"]
# test["tenure_squared"] = test["tenure"] * test["tenure"]

# train["tenure_cubed"] = train["tenure"] * train["tenure"] * train["tenure"]
# test["tenure_cubed"] = test["tenure"] * test["tenure"] * test["tenure"]

train["Churn"] = train["Churn"].map({"No": 0, "Yes": 1})

X = train.drop("id",axis=1)
y = X.pop("Churn")

X_test = test.drop("id",axis=1)

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "n_estimators":300,
    #"learning_rate":0.01,
    #"max_depth":4,
    "random_state":42,
}
model = LGBMClassifier(**params)


X_enc = X.copy()

ohe_features = ["Contract","PaymentMethod"]

for col in ohe_features:
    X_enc[col] = X_enc[col].astype("category")
    X_test[col] = X_test[col].astype("category")

model.fit(X_enc, y)
#---------------------------------------------


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_enc)


importance = np.abs(shap_values).mean(axis=0)

shap_importance = pd.Series(
    importance,
    index=X_enc.columns
).sort_values(ascending=False)

#print(shap_importance)

#------------------------

for col in X_enc.select_dtypes(include="category"):
    le = LabelEncoder()
    X_enc[col] = le.fit_transform(X_enc[col])


mi = mutual_info_classif(X_enc, y , random_state=42)

mi_scores = pd.Series(mi, index=X_enc.columns)
mi_scores = mi_scores.sort_values(ascending=False)

# print(mi_scores)


#------------------------

feature_scores = pd.concat(
    [mi_scores, shap_importance],
    axis=1
)

feature_scores.columns = ["MI", "SHAP"]

print(feature_scores)
print("-"*40)
#------------------------

redundant = feature_scores[
    (feature_scores["MI"] > 0.05) &
    (feature_scores["SHAP"] < 0.02)
]

print("redundants:")
print(redundant)
print("-"*40)

#------------------------
useless = feature_scores[
    (feature_scores["MI"] < 0.02) &
    (feature_scores["SHAP"] < 0.01)
]

print("useless:")
print(useless)
print("-"*40)

#------------------------

corr = X_enc.corr().abs()

upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)

drop_cols = [
    column for column in upper.columns
    if any(upper[column] > 0.95)
]

print("drop cols")
print(drop_cols)
print("-"*40)

#-------------------------

# baseline_scores = cross_val_score(
#     model,
#     X_enc, y,
#     cv=5,
#     scoring="roc_auc",
#     n_jobs=-1
# )

# print(baseline_scores.mean())

# #-------------------------
# X_enc_dropped = X_enc.drop(["Partner","gender"], axis=1)

# pruning_scores = cross_val_score(
#     model,
#     X_enc_dropped, y,
#     cv=5,
#     scoring="roc_auc",
#     n_jobs=-1
# )

# print(pruning_scores.mean())

#-------------------------