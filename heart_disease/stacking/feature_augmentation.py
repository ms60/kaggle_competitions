from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

oof_X = pd.read_csv("meta_X_1.csv")
oof_test = pd.read_csv("meta_test_1.csv")

X["lgbm"] = oof_X["lgbm"]
test["lgbm"] = oof_test["lgbm"]



X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075, shuffle=True ,stratify=y)

def objective(trial):

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",      
        "n_estimators": trial.suggest_int("n_estimators", 300, 7000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "verbosity": 0,
        #"early_stopping_rounds":100,

    }

    model = XGBClassifier(**params)
    model.fit(X_train,y_train)

    #y_preds = model.predict(X_valid_proc)
    y_proba = model.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, y_proba)
    
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=75)