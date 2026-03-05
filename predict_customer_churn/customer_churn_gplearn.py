from lightgbm import LGBMClassifier, LGBMRegressor
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from gplearn.genetic import SymbolicTransformer

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


train["Churn"] = train["Churn"].map({"No": 0, "Yes": 1})

X = train.drop("id",axis=1)
y = X.pop("Churn")

X_test = test.drop("id",axis=1)


X_train , X_valid , y_train , y_valid = train_test_split(X[["MonthlyCharges","TotalCharges","tenure"]] , y , test_size=0.075 , random_state=42 , stratify=y)



gp = SymbolicTransformer(
    metric='log loss',
    generations=20,
    population_size=1000,
    hall_of_fame=10,
    n_components=1,
    function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos' , 'sqrt' , 'log' , 'inv'),
    parsimony_coefficient=0.01,
    max_samples=0.9,
    random_state=42,
    verbose=1
)

gp.fit(X_train, y_train)

gp_features_train = gp.transform(X_train)
gp_features_valid = gp.transform(X_valid)
gp_features_test = gp.transform(X_test)


gp_feature_names = [f"gp_{i}" for i in range(gp_features_train.shape[1])]

gp_train_df = pd.DataFrame(gp_features_train, columns=gp_feature_names)
gp_valid_df = pd.DataFrame(gp_features_valid, columns=gp_feature_names)
gp_test_df = pd.DataFrame(gp_features_test, columns=gp_feature_names)




best_params =  {'boosting_type': 'gbdt', 'n_estimators': 3686, 'learning_rate': 0.042233987505403366, 'num_leaves': 135, 'max_depth': 5, 'min_child_samples': 76, 'min_child_weight': 0.17465047574971193, 'min_split_gain': 0.030362576690836283, 'subsample': 0.46027777923464497, 'colsample_bytree': 0.4420788366501833, 'reg_alpha': 0.006730096397606108, 'reg_lambda': 1.0041760611014332}
best_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,

})

best_model = LGBMClassifier(**best_params)
best_model.fit(X_train, y_train)

y_proba_base = best_model.predict_proba(X_valid)[:, 1]
base_score = roc_auc_score(y_valid, y_proba_base)

print(base_score)



X_train_aug = pd.concat([X_train.reset_index(drop=True),
                         gp_train_df.reset_index(drop=True)], axis=1)

X_valid_aug = pd.concat([X_valid.reset_index(drop=True),
                        gp_valid_df.reset_index(drop=True)], axis=1)

model_aug = LGBMClassifier(**best_params)
model_aug.fit(X_train_aug, y_train)

y_proba_aug = model_aug.predict(X_valid_aug)
score_aug = roc_auc_score(y_valid, y_proba_aug)

print(score_aug)















