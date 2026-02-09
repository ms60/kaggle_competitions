from category_encoders import TargetEncoder
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import clone
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score, train_test_split, KFold , StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import mean_absolute_error, r2_score , accuracy_score , classification_report ,  precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder , StandardScaler 


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})




X["f1"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Sex"] == 1)
X["f2"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f3"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f4"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)

X["f5"] =   (X["Sex"] == 1) & (X["Chest pain type"] == 3) & (X["EKG results"] == 2)

X["f6"] = (X["Thallium"] == 3) & (X["Age"] < 53.00)


test["f1"] = (test["Age"] < 37.00) & (test["Thallium"] == 7) & (test["Sex"] == 1)
test["f2"] = (test["Age"] < 37.00) & (test["Thallium"] == 7) & (test["Exercise angina"] == 1)
test["f3"] = (test["Age"] > 69.00) & (test["Thallium"] == 7) & (test["Exercise angina"] == 1)
test["f4"] = (test["Age"] > 69.00) & (test["Thallium"] == 7) & (test["FBS over 120"] == 1)

test["f5"] =   (test["Sex"] == 1) & (test["Chest pain type"] == 3) & (test["EKG results"] == 2)

test["f6"] = (test["Thallium"] == 3) & (test["Age"] < 53.00)

X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075, shuffle=True ,stratify=y)

def objective(trial):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': trial.suggest_int('n_estimators',6000,8500),# 7000,#8000, # 6405
        'learning_rate': 0.03,
        'num_leaves': trial.suggest_int('num_leaves', 60, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 180),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
        'reg_alpha': 3.9,
        'reg_lambda': 0.26,
        'random_state': 42,
        'verbosity': -1
    }

    # --- CV setup ---
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    best_iters = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='auc',
            callbacks=[
                early_stopping(200),
                log_evaluation(0)  # sessiz mod
            ]
        )

        preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        aucs.append(auc)
        best_iters.append(model.best_iteration_)

    # --- CV ortalaması ---
    mean_auc = np.mean(aucs)
    mean_best_iter = int(np.mean(best_iters))
    trial.set_user_attr("mean_best_iter", mean_best_iter)

    return mean_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best params:", study.best_params)
print("Best AUC:", study.best_value)
print("Mean best_iter:", study.best_trial.user_attrs["mean_best_iter"])