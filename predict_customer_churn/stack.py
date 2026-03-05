from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from hill_climbing import Climber
from hill_climbing import ClimberCV

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

#-------------------
#X_raw

X_raw = train.drop("id",axis=1)
y = X_raw.pop("Churn")

X_test_raw = test.drop("id",axis=1)

ohe_features = ["Contract","PaymentMethod"]
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]


for col in ohe_features:
    X_raw[col] = X_raw[col].astype("category")
    X_test_raw[col] = X_test_raw[col].astype("category")


X_raw_params = {'boosting_type': 'gbdt', 'n_estimators': 3004, 'learning_rate': 0.11213396940737012, 'num_leaves': 46, 'max_depth': 5, 'min_child_samples': 246, 'min_child_weight': 0.03656034336099059, 'min_split_gain': 0.0842741427392219, 'subsample': 0.13176912082933098, 'colsample_bytree': 0.3555874786685853, 'reg_alpha': 0.3143653644797928, 'reg_lambda': 3.668487003914375}
X_raw_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,
})


#----------------------------------------------------
# X_numeric
X_numeric = train[["tenure", "MonthlyCharges", "TotalCharges"]]
X_numeric_test = test[["tenure", "MonthlyCharges", "TotalCharges"]]

X_numeric_params = {'boosting_type': 'gbdt', 'n_estimators': 2704, 'learning_rate': 0.055137104771518314, 'num_leaves': 83, 'max_depth': 4, 'min_child_samples': 179, 'min_child_weight': 7.726222831043295, 'min_split_gain': 0.13328858192466775, 'subsample': 0.478167910829483, 'colsample_bytree': 0.9997339343474747, 'reg_alpha': 1.0818184161942739, 'reg_lambda': 0.7744567974883169}
X_numeric_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,
})


#----------------------------------------------------
# X_numeric_log1p
X_numeric_log1p = train[["tenure", "MonthlyCharges", "TotalCharges"]]
X_numeric_log1p_test = test[["tenure", "MonthlyCharges", "TotalCharges"]]

for col in numeric_features:
    X_numeric_log1p[col] = np.log1p(X_numeric_log1p[col])
    X_numeric_log1p_test[col] = np.log1p(X_numeric_log1p_test[col])

X_numeric_log1p_params = {'boosting_type': 'gbdt', 'n_estimators': 4188, 'learning_rate': 0.153121558468246, 'num_leaves': 123, 'max_depth': 4, 'min_child_samples': 199, 'min_child_weight': 5.342236511883262, 'min_split_gain': 0.12234672850570662, 'subsample': 0.31185689615356715, 'colsample_bytree': 0.870585214152224, 'reg_alpha': 3.1053798047020487, 'reg_lambda': 0.0010023841853419992}
X_numeric_log1p_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,
})



#--------------------------------
#X_numeric_squared
X_numeric_squared = train[["tenure", "MonthlyCharges", "TotalCharges"]]
X_numeric_squared_test = test[["tenure", "MonthlyCharges", "TotalCharges"]]

for col in numeric_features:
    X_numeric_squared[col] = X_numeric_squared[col] ** 2
    X_numeric_squared_test[col] = X_numeric_squared_test[col] ** 2

X_numeric_squared_params = {'boosting_type': 'gbdt', 'n_estimators': 4331, 'learning_rate': 0.014798206326399829, 'num_leaves': 62, 'max_depth': 5, 'min_child_samples': 16, 'min_child_weight': 0.04891876261664777, 'min_split_gain': 0.12401711352272889, 'subsample': 0.4127479655363408, 'colsample_bytree': 0.999527922343225, 'reg_alpha': 0.0025363231517635874, 'reg_lambda': 1.1267862286281594}
X_numeric_squared_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,
})

#-----------------------
#X_categorical_1
X_categorical_1 = train[["gender", "Partner", "Dependents", "PhoneService","InternetService","PaperlessBilling"]]
X_categorical_1_test = test[["gender", "Partner", "Dependents", "PhoneService","InternetService","PaperlessBilling"]]

X_categorical_1_params = {'boosting_type': 'gbdt', 'n_estimators': 2604, 'learning_rate': 0.17200550901374795, 'num_leaves': 231, 'max_depth': 4, 'min_child_samples': 14, 'min_child_weight': 0.18826101365135506, 'min_split_gain': 0.7822493368572241, 'subsample': 0.585304336205543, 'colsample_bytree': 0.696693948944469, 'reg_alpha': 0.20713864647629132, 'reg_lambda': 0.024113861411454773}
X_categorical_1_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,
})



#-----------------------
#X_categorical_2
X_categorical_2 = train[["gender", "Partner", "Dependents", "PhoneService","InternetService","PaperlessBilling" , "Contract","PaymentMethod"]]
X_categorical_2_test = test[["gender", "Partner", "Dependents", "PhoneService","InternetService","PaperlessBilling" , "Contract","PaymentMethod"]]

for col in ohe_features:
    X_categorical_2[col] = X_categorical_2[col].astype("category")
    X_categorical_2_test[col] = X_categorical_2_test[col].astype("category")

X_categorical_2_params = {'boosting_type': 'gbdt', 'n_estimators': 1739, 'learning_rate': 0.22455264085144352, 'num_leaves': 176, 'max_depth': 4, 'min_child_samples': 18, 'min_child_weight': 0.1476946551673337, 'min_split_gain': 0.028739947148094337, 'subsample': 0.5380179444922599, 'colsample_bytree': 0.5099821637068387, 'reg_alpha': 0.09826403490705252, 'reg_lambda': 0.0026413431684346885}
X_categorical_2_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,
})



#-----------------------
#X_categorical_3
X_categorical_3 =train[["gender", "Partner", "Dependents", "PhoneService","InternetService","PaperlessBilling" , "MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies", "Contract","PaymentMethod"]]
X_categorical_3_test = test[["gender", "Partner", "Dependents", "PhoneService","InternetService","PaperlessBilling" , "MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies", "Contract","PaymentMethod"]]

for col in ohe_features:
    X_categorical_3[col] = X_categorical_3[col].astype("category")
    X_categorical_3_test[col] = X_categorical_3_test[col].astype("category")

X_categorical_3_params = {'boosting_type': 'gbdt', 'n_estimators': 3142, 'learning_rate': 0.008760352192201339, 'num_leaves': 68, 'max_depth': 5, 'min_child_samples': 67, 'min_child_weight': 0.03660133957906329, 'min_split_gain': 0.004243778505248947, 'subsample': 0.8750827604050337, 'colsample_bytree': 0.31555126365393316, 'reg_alpha': 0.014430924191382435, 'reg_lambda': 0.37003698142244373}
X_categorical_3_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,
})


#-----------------------
#X_1
X_1 = train.drop(["id","Churn"],axis=1)
X_1_test = test.drop("id",axis=1)


X_1["InternetQuality"] =   (X_1["OnlineSecurity"] + X_1["OnlineBackup"] + X_1["DeviceProtection"] + X_1["TechSupport"] + X_1["StreamingTV"] + X_1["StreamingMovies"] )
X_1_test["InternetQuality"] =  X_1_test["OnlineSecurity"] + X_1_test["OnlineBackup"] + X_1_test["DeviceProtection"] + X_1_test["TechSupport"] + X_1_test["StreamingTV"] + X_1_test["StreamingMovies"] 

X_1["TotalCharges_over_tenure"] = X_1["TotalCharges"] / X_1["tenure"]
X_1_test["TotalCharges_over_tenure"] = X_1_test["TotalCharges"] / X_1_test["tenure"]

X_1["Security"] = X_1["OnlineSecurity"] * X_1["DeviceProtection"] * X_1["TechSupport"]
X_1_test["Security"] = X_1_test["OnlineSecurity"] * X_1_test["DeviceProtection"] * X_1_test["TechSupport"]

X_1["ExtraFeatures"] = X_1["StreamingTV"] * X_1["StreamingMovies"]
X_1_test["ExtraFeatures"] = X_1_test["StreamingTV"] * X_1_test["StreamingMovies"]

for col in ohe_features:
    X_1[col] = X_1[col].astype("category")
    X_1_test[col] = X_1_test[col].astype("category")

X_1_params = {'boosting_type': 'gbdt', 'n_estimators': 3686, 'learning_rate': 0.042233987505403366, 'num_leaves': 135, 'max_depth': 5, 'min_child_samples': 76, 'min_child_weight': 0.17465047574971193, 'min_split_gain': 0.030362576690836283, 'subsample': 0.46027777923464497, 'colsample_bytree': 0.4420788366501833, 'reg_alpha': 0.006730096397606108, 'reg_lambda': 1.0041760611014332}
X_1_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,

})

#-----------------------
#X_1_new_features

X_1_new_features = X_1[["InternetQuality","TotalCharges_over_tenure","Security","ExtraFeatures"]]
X_1_new_features_test = X_1_test[["InternetQuality","TotalCharges_over_tenure","Security","ExtraFeatures"]]

X_1_new_features_params = {'boosting_type': 'gbdt', 'n_estimators': 3700, 'learning_rate': 0.006235345028478144, 'num_leaves': 245, 'max_depth': 5, 'min_child_samples': 10, 'min_child_weight': 0.0063483910293922535, 'min_split_gain': 0.0006609387390904588, 'subsample': 0.8703628013923899, 'colsample_bytree': 0.9050049046498958, 'reg_alpha': 0.0012137665256284436, 'reg_lambda': 0.49480279633725355}
X_1_new_features_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,

})



#-----------------------
#X_2
X_2 = X_1.copy()
X_2_test = X_1_test.copy()


totalCharges_residual = pd.read_csv("./stack/TotalCharges_residual.csv")
totalCharges_residual_test = pd.read_csv("./stack/TotalCharges_residual_test.csv")

X_2["TotalCharges_residual"] = totalCharges_residual["TotalCharges_residual"]
X_2_test["TotalCharges_residual"] = totalCharges_residual_test["TotalCharges_residual"]

X_2_params = {'boosting_type': 'gbdt', 'n_estimators': 4910, 'learning_rate': 0.01458978383427039, 'num_leaves': 45, 'max_depth': 5, 'min_child_samples': 11, 'min_child_weight': 4.697798759563624, 'min_split_gain': 0.08215945522822365, 'subsample': 0.1826636157326016, 'colsample_bytree': 0.3004733863428752, 'reg_alpha': 0.0027709022383171325, 'reg_lambda': 0.17550013459955266}
X_2_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,

})

#-------------------------------
#X_3

X_3 = train.drop(["id","Churn"],axis=1)
X_3_test = test.drop("id",axis=1)


X_3["InternetQuality"] =   X_3["OnlineSecurity"] + X_3["OnlineBackup"] + X_3["DeviceProtection"] + X_3["TechSupport"] + X_3["StreamingTV"] + X_3["StreamingMovies"]
X_3_test["InternetQuality"] =  X_3_test["OnlineSecurity"] + X_3_test["OnlineBackup"] + X_3_test["DeviceProtection"] + X_3_test["TechSupport"] + X_3_test["StreamingTV"] + X_3_test["StreamingMovies"] 

X_3["Security"] = X_3["OnlineSecurity"] * X_3["DeviceProtection"] * X_3["TechSupport"]
X_3_test["Security"] = X_3_test["OnlineSecurity"] * X_3_test["DeviceProtection"] * X_3_test["TechSupport"]

X_3['Expected_TotalCharges'] = X_3['tenure'] * X_3['MonthlyCharges']
X_3_test['Expected_TotalCharges'] = X_3_test['tenure'] * X_3_test['MonthlyCharges']

X_3['Charges_Difference'] = X_3['TotalCharges'] - X_3['Expected_TotalCharges']
X_3_test['Charges_Difference'] = X_3_test['TotalCharges'] - X_3_test['Expected_TotalCharges']

X_3['Cost_Per_Month_Tenure'] = X_3['TotalCharges'] / (X_3['tenure'] + 1)
X_3_test['Cost_Per_Month_Tenure'] = X_3_test['TotalCharges'] / (X_3_test['tenure'] + 1)

services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
X_3['Total_Services'] = X_3[services].sum(axis=1)
X_3_test['Total_Services'] = X_3_test[services].sum(axis=1)


X_3['Avg_Cost_Per_Service'] = X_3['MonthlyCharges'] / (X_3['Total_Services'] + 1)
X_3_test['Avg_Cost_Per_Service'] = X_3_test['MonthlyCharges'] / (X_3_test['Total_Services'] + 1)



contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
X_3['Contract_Months'] = X_3['Contract'].map(contract_map).astype(float).fillna(1.0)
X_3_test['Contract_Months'] = X_3_test['Contract'].map(contract_map).astype(float).fillna(1.0)


X_3['Tenure_Contract_Ratio'] = X_3['tenure'] / X_3['Contract_Months']
X_3_test['Tenure_Contract_Ratio'] = X_3_test['tenure'] / X_3_test['Contract_Months']

X_3["Is_Safe_Customer"] = X_3["OnlineSecurity"] * X_3["TechSupport"]
X_3_test["Is_Safe_Customer"] = X_3_test["OnlineSecurity"] * X_3_test["TechSupport"]


X_3['Is_Auto_Payment'] = X_3['PaymentMethod'].astype(str).str.contains('automatic', case=False).astype(int)
X_3_test['Is_Auto_Payment'] = X_3_test['PaymentMethod'].astype(str).str.contains('automatic', case=False).astype(int)

X_3["Is_Streaming_Fan"] = X_3["StreamingMovies"] * X_3["StreamingTV"]
X_3_test["Is_Streaming_Fan"] = X_3_test["StreamingMovies"] * X_3_test["StreamingTV"]

X_3['Monthly_per_Contract_Month'] = X_3['MonthlyCharges'] / X_3['Contract_Months']
X_3_test['Monthly_per_Contract_Month'] = X_3_test['MonthlyCharges'] / X_3_test['Contract_Months']

for col in ohe_features:
    X_3[col] = X_3[col].astype("category")
    X_3_test[col] = X_3_test[col].astype("category")

X_3_params = best_params = {'boosting_type': 'gbdt', 'n_estimators': 4967, 'learning_rate': 0.00982101577044508, 'num_leaves': 171, 'max_depth': 6, 'min_child_samples': 21, 'min_child_weight': 7.6187438604004845, 'min_split_gain': 0.19697909076020012, 'subsample': 0.7307448230595652, 'colsample_bytree': 0.18611740562769047, 'reg_alpha': 0.012421016741057854, 'reg_lambda': 0.4164730346740021}
X_3_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,

})

#----------------------------------------
#X_3_new_features
X_3_new_features = X_3[["InternetQuality","Security","Expected_TotalCharges","Charges_Difference","Cost_Per_Month_Tenure","Total_Services","Avg_Cost_Per_Service","Tenure_Contract_Ratio","Is_Safe_Customer","Is_Auto_Payment","Is_Streaming_Fan","Monthly_per_Contract_Month"]]
X_3_new_features_test = X_3_test[["InternetQuality","Security","Expected_TotalCharges","Charges_Difference","Cost_Per_Month_Tenure","Total_Services","Avg_Cost_Per_Service","Tenure_Contract_Ratio","Is_Safe_Customer","Is_Auto_Payment","Is_Streaming_Fan","Monthly_per_Contract_Month"]]

X_3_new_features_params = {'boosting_type': 'gbdt', 'n_estimators': 3984, 'learning_rate': 0.019288676774736623, 'num_leaves': 253, 'max_depth': 5, 'min_child_samples': 23, 'min_child_weight': 1.9425762510360471, 'min_split_gain': 0.05743833235201379, 'subsample': 0.22685585612377912, 'colsample_bytree': 0.46273100031603837, 'reg_alpha': 1.6289821836286142, 'reg_lambda': 0.00161714978446964}
X_3_new_features_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,

})

#--------------------------------------
# elastic - ridge - lasso with different parameters

#--------------------------------------
# DL with different parameters

#--------------------------------------
# some other tree models with different parameters

#--------------------------------------



#---------------------------------------

#run optuna for each

# DATASET = X_3_new_features

# X_train , X_valid , y_train , y_valid = train_test_split(DATASET , y , test_size=0.075 , random_state=42 , stratify=y)
# X_train_2 , X_valid_2 , y_train_2 , y_valid_2 = train_test_split(DATASET , y , test_size=0.075 , random_state=60 , stratify=y)


# def objective(trial):
#     params= {
#     "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]),
#     "objective": "binary",
#     "metric": "auc",
#     "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
#     "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5,log=True),
#     "num_leaves": trial.suggest_int("num_leaves", 10, 256),
#     "max_depth": trial.suggest_int("max_depth", 3, 5),#,3,#trial.suggest_int("max_depth", 3, 8),
#     "min_child_samples": trial.suggest_int("min_child_samples", 10, 300,log=True),
#     "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
#     "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
#     "subsample": trial.suggest_float("subsample", 0.1, 1.0),
#     #"subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
#     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0 , log=True),
#     "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0,log=True),
#     "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0,log=True),
#     #"max_bin": trial.suggest_int("max_bin", 64, 512),
#     "random_state": 42,
#     "verbosity": -1,
#     }



#     model = LGBMClassifier(**params)
#     model.fit(X_train,y_train)

#     #y_preds = model.predict(X_valid_proc)
#     y_proba = model.predict_proba(X_valid)[:, 1]

#     score = roc_auc_score(y_valid, y_proba)

#     model2 = LGBMClassifier(**params)
#     model2.fit(X_train_2,y_train_2)
#     y_proba2 = model2.predict_proba(X_valid_2)[:, 1]
#     score2 = roc_auc_score(y_valid_2, y_proba2)

    
#     return  (score + score2) / 2 

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=60)

# print("Best ROC AUC:", study.best_value)
# print("Best params:", study.best_params)

# best_params = study.best_params


#---------------------
# generate oofs

# oof_preds = np.zeros(len(X_1), dtype=float) 
# test_preds = np.zeros(len(X_1_test), dtype=float) 

# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# for fold, (train_idx, valid_idx) in enumerate(skf.split(X_3_new_features, y)):

#     print(f"Fold {fold+1}")

#     X_train, X_valid = X_3_new_features.iloc[train_idx], X_3_new_features.iloc[valid_idx]
#     y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

#     model = LGBMClassifier(**X_3_new_features_params)
#     model.fit(X_train,y_train)

#     oof_preds[valid_idx] = model.predict_proba(X_valid)[:,1]
#     test_preds += model.predict_proba(X_3_new_features_test)[:,1] / skf.n_splits


# oof_X = pd.DataFrame({"oof_pred": oof_preds})
# oof_test = pd.DataFrame({"oof_pred_test": test_preds})

# oof_X.to_csv("./stack/X_3_new_features_oof.csv",index=False)
# oof_test.to_csv("./stack/X_3_new_features_oof_test.csv",index=False)


#--------------------------------------
# stacking

X_raw_oof = pd.read_csv("./stack/X_raw_oof.csv")
X_numeric_oof = pd.read_csv("./stack/X_numeric_oof.csv")
X_1_oof = pd.read_csv("./stack/X_1_oof.csv")
X_1_new_features_oof = pd.read_csv("./stack/X_1_new_features_oof.csv")
X_categorical_1_oof = pd.read_csv("./stack/X_categorical_1_oof.csv")
X_categorical_2_oof = pd.read_csv("./stack/X_categorical_2_oof.csv")
X_categorical_3_oof = pd.read_csv("./stack/X_categorical_3_oof.csv")
X_2_oof = pd.read_csv("./stack/X2_oof.csv")
X_3_oof = pd.read_csv("./stack/X3_oof.csv")
X_3_new_features_oof = pd.read_csv("./stack/X_3_new_features_oof.csv")


X_raw_oof_test = pd.read_csv("./stack/X_raw_oof_test.csv")
X_numeric_oof_test = pd.read_csv("./stack/X_numeric_oof_test.csv")
X_1_oof_test = pd.read_csv("./stack/X_1_oof_test.csv")
X_1_new_features_oof_test = pd.read_csv("./stack/X_1_new_features_oof_test.csv")
X_categorical_1_oof_test = pd.read_csv("./stack/X_categorical_1_oof_test.csv")
X_categorical_2_oof_test = pd.read_csv("./stack/X_categorical_2_oof_test.csv")
X_categorical_3_oof_test = pd.read_csv("./stack/X_categorical_3_oof_test.csv")
X_2_oof_test = pd.read_csv("./stack/X2_oof_test.csv")
X_3_oof_test = pd.read_csv("./stack/X3_oof_test.csv")
X_3_new_features_oof_test = pd.read_csv("./stack/X_3_new_features_oof_test.csv")





X_oof_total = pd.concat([X_raw_oof,X_numeric_oof,X_1_oof,X_1_new_features_oof,X_categorical_1_oof,X_categorical_2_oof,X_categorical_3_oof,X_2_oof,X_3_oof,X_3_new_features_oof],axis=1)
X_oof_total.columns = ["X_raw_oof","X_numeric_oof","X_1_oof","X_1_new_features_oof","X_categorical_1_oof","X_categorical_2_oof","X_categorical_3_oof","X_2_oof" , "X_3_oof","X_3_new_features_oof"]

X_oof_test_total = pd.concat([X_raw_oof_test,X_numeric_oof_test,X_1_oof_test,X_1_new_features_oof_test,X_categorical_1_oof_test,X_categorical_2_oof_test,X_categorical_3_oof_test,X_2_oof_test,X_3_oof_test,X_3_new_features_oof_test],axis=1)
X_oof_test_total.columns = ["X_raw_oof","X_numeric_oof","X_1_oof","X_1_new_features_oof","X_categorical_1_oof","X_categorical_2_oof","X_categorical_3_oof","X_2_oof" , "X_3_oof","X_3_new_features_oof"]#["X_raw_oof_test","X_numeric_oof_test","X_1_oof_test","X_1_new_features_oof_test","X_categorical_1_oof_test","X_categorical_2_oof_test","X_categorical_3_oof_test"]



print(X_oof_total)

print(X_oof_total.corr())

#----------------------
# meta model

# meta_model = LogisticRegression(
#     penalty="l2",        # Ridge
#     C=1.0,               # Regularization gücü (küçük C = daha güçlü)
#     solver="lbfgs",      # default ve stabil
#     max_iter=1000,
#     n_jobs=-1
# )


X_train , X_valid , y_train , y_valid = train_test_split(X_oof_total , y , test_size=0.075 , random_state=42 , stratify=y)
X_train_2 , X_valid_2 , y_train_2 , y_valid_2 = train_test_split(X_oof_total , y , test_size=0.075 , random_state=60 , stratify=y)



climber_cv = ClimberCV(
    objective="maximize",
    eval_metric=roc_auc_score,
    cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
)

climber_cv.fit(X_oof_total, y)
final_test = climber_cv.predict(X_oof_test_total)

result = pd.DataFrame({ "id": test["id"], "Churn": final_test })
result.to_csv("result.csv",index=False)