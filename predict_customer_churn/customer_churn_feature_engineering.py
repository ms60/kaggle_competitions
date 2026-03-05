from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from matplotlib import pyplot as plt
import seaborn as sns

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

train["InternetQuality"] =   (train["OnlineSecurity"] + train["OnlineBackup"] + train["DeviceProtection"] + train["TechSupport"] + train["StreamingTV"] + train["StreamingMovies"] )
test["InternetQuality"] =  test["OnlineSecurity"] + test["OnlineBackup"] + test["DeviceProtection"] + test["TechSupport"] + test["StreamingTV"] + test["StreamingMovies"] 

train["TotalCharges_over_tenure"] = train["TotalCharges"] / train["tenure"]
test["TotalCharges_over_tenure"] = test["TotalCharges"] / test["tenure"]

train["Security"] = train["OnlineSecurity"] * train["DeviceProtection"] * train["TechSupport"]
test["Security"] = test["OnlineSecurity"] * test["DeviceProtection"] * test["TechSupport"]

train["ExtraFeatures"] = train["StreamingTV"] * train["StreamingMovies"]
test["ExtraFeatures"] = test["StreamingTV"] * test["StreamingMovies"]





# train["tenure_squared"] = train["tenure"] ** 2
# test["tenure_squared"] = test["tenure"] ** 2




train["Churn"] = train["Churn"].map({"No": 0, "Yes": 1})

X = train.drop("id",axis=1)
y = X.pop("Churn")

X_test = test.drop("id",axis=1)


totalCharges_residual = pd.read_csv("./stack/TotalCharges_residual.csv")
totalCharges_residual_test = pd.read_csv("./stack/TotalCharges_residual_test.csv")

X["TotalCharges_residual"] = totalCharges_residual["TotalCharges_residual"]
X_test["TotalCharges_residual"] = totalCharges_residual_test["TotalCharges_residual"]


numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
ohe_features = ["Contract","PaymentMethod"]


for col in ohe_features:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")



#best_params =  {"colsample_bytree":0.8,"subsample":0.8,"max_depth":3,'boosting_type': 'gbdt','n_estimators': 1832, 'learning_rate': 0.14454389953357547, 'num_leaves': 84, 'reg_alpha': 2.5671620451305492, 'reg_lambda': 0.009180210576377256}
#best_params = {'boosting_type': 'gbdt', 'n_estimators': 3044, 'learning_rate': 0.08689656825820848, 'num_leaves': 210, 'max_depth': 4, 'min_child_samples': 18, 'min_child_weight': 0.011974187303327171, 'min_split_gain': 0.07422986122739264, 'subsample': 0.35954490266830175, 'colsample_bytree': 0.5024304816412929, 'reg_alpha': 0.034860128015873, 'reg_lambda': 0.05003006774124102}
#best_params = {'boosting_type': 'gbdt', 'n_estimators': 3207, 'learning_rate': 0.07931040793346719, 'num_leaves': 84, 'max_depth': 5, 'min_child_samples': 85, 'min_child_weight': 0.22755937045227428, 'min_split_gain': 0.061571078804342816, 'subsample': 0.7353569513590934, 'colsample_bytree': 0.5335276884646094, 'reg_alpha': 0.6154086150021465, 'reg_lambda': 0.3903499106248194}

#best_params =  {'boosting_type': 'gbdt', 'n_estimators': 3686, 'learning_rate': 0.042233987505403366, 'num_leaves': 135, 'max_depth': 5, 'min_child_samples': 76, 'min_child_weight': 0.17465047574971193, 'min_split_gain': 0.030362576690836283, 'subsample': 0.46027777923464497, 'colsample_bytree': 0.4420788366501833, 'reg_alpha': 0.006730096397606108, 'reg_lambda': 1.0041760611014332}
best_params = {'boosting_type': 'gbdt', 'n_estimators': 4910, 'learning_rate': 0.01458978383427039, 'num_leaves': 45, 'max_depth': 5, 'min_child_samples': 11, 'min_child_weight': 4.697798759563624, 'min_split_gain': 0.08215945522822365, 'subsample': 0.1826636157326016, 'colsample_bytree': 0.3004733863428752, 'reg_alpha': 0.0027709022383171325, 'reg_lambda': 0.17550013459955266}


best_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,

})




best_model = LGBMClassifier(**best_params)
best_model.fit(X,y)

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# scores = cross_val_score(best_model, X, y, cv=skf, scoring="roc_auc")
# print(scores)
# print(scores.mean())
# print(scores.std())
# print(scores)

feature_imp = pd.DataFrame(sorted(zip(best_model.feature_importances_,X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

# explainer = shap.TreeExplainer(best_model)
# shap_values = explainer(X)

# shap.plots.beeswarm(shap_values)

#baseline
# [0.91622643 0.91733608 0.91674406 0.91770254 0.91505162]
# 0.916612145469113
# 0.0009289200269665468