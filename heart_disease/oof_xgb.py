import optuna
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

X["ind_1"] = (X["Age"] > 40 ) & ( X["Exercise angina"] )
X["ind_2"] = (X["Age"] > 50) &  (X["FBS over 120"] )
X["ind_3"] = ( X["Exercise angina"] ) &  (X["ST depression"] > 1 )
X["ind_4"] = ( X["Exercise angina"] ) & (X["FBS over 120"] )

X["ind_5"] = (X["ind_4"] ) & ( X["EKG results"] > 0 )

X["ind_6"] = X["ind_3"] &  X["ind_4"] & (X["Thallium"] == 3)

X["ind_7"] = ( X["EKG results"] > 0 ) & ( X["Chest pain type"]==3 )
X["ind_8"] = ( X["EKG results"] > 0 ) & ( X["Chest pain type"]==2 )

X["ind_9"] =  ( X["Number of vessels fluro"] > 0 ) & (X["Age"] < 40) 

X["ind_10"] =  (X["Sex"] ==1 )  & (X["Age"] > 40 ) & ( X["Cholesterol"] > 300 )
X["ind_11"] =  (X["Sex"] ==1 )  & (X["Age"] > 40 ) & ( X["BP"] > 180 )

X["ind_12"] =   (X["ST depression"] > 2.5 ) & ( X["Slope of ST"] >= 2 ) & (X["Age"] > 50   )


for i in range(1,13):
    X["ind_"+str(i)] = X["ind_"+str(i)].astype(int) 

    test["ind_1"] = (test["Age"] > 40 ) & ( test["Exercise angina"] )
test["ind_2"] = (test["Age"] > 50) &  (test["FBS over 120"] )
test["ind_3"] = ( test["Exercise angina"] ) &  (test["ST depression"] > 1 )
test["ind_4"] = ( test["Exercise angina"] ) & (test["FBS over 120"] )

test["ind_5"] = (test["ind_4"] ) & ( test["EKG results"] > 0 )

test["ind_6"] = test["ind_3"] &  test["ind_4"] & (test["Thallium"] == 3)

test["ind_7"] = ( test["EKG results"] > 0 ) & ( test["Chest pain type"]==3 )
test["ind_8"] = ( test["EKG results"] > 0 ) & ( test["Chest pain type"]==2 )

test["ind_9"] =  ( test["Number of vessels fluro"] > 0 ) & (test["Age"] < 40) 

test["ind_10"] =  (test["Sex"] ==1 )  & (test["Age"] > 40 ) & ( test["Cholesterol"] > 300 )
test["ind_11"] =  (test["Sex"] ==1 )  & (test["Age"] > 40 ) & ( test["BP"] > 180 )

test["ind_12"] =   (test["ST depression"] > 2.5 ) & ( test["Slope of ST"] >= 2 ) & (test["Age"] > 50   )

for i in range(1,13):
    test["ind_"+str(i)] = test["ind_"+str(i)].astype(int) 




def objective(trial):

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",      # GPU varsa "gpu_hist"
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds":100,

    }

    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr , 
                  eval_set=[(X_val, y_val)],

                   )

        oof[val_idx] = model.predict_proba(X_val)[:, 1]

    return roc_auc_score(y, oof)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# print("Best OOF AUC:", study.best_value)
# print("Best params:", study.best_params)

# best_params = study.best_params
# best_params.update({
#     "objective": "binary:logistic",
#     "eval_metric": "auc",
#     "tree_method": "hist",
#     "random_state": 42,
#     "verbosity": 0
# })

best_params = {'n_estimators': 3409, 'max_depth': 4, 'num_leaves': 15, 'min_child_samples': 99, 'learning_rate': 0.1684214301847516, 'subsample': 0.9772643991731278, 'colsample_bytree': 0.9453450226985814, 'reg_alpha': 0.5577589164067142, 'reg_lambda': 0.3741296432515172}

oof_pred_xgb = np.zeros(len(X))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = XGBClassifier(**best_params)
    model.fit(X_tr, y_tr)

    oof_pred_xgb[val_idx] = model.predict_proba(X_val)[:, 1]

    fold_auc = roc_auc_score(y_val, oof_pred_xgb[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

print("Final OOF XGB AUC:", roc_auc_score(y, oof_pred_xgb))

pd.Series(
    oof_pred_xgb,
    name="oof_pred_xgb"
).to_csv("oof_pred_xgb.csv", index=False)

# --------------------------------
# Test OOF üret
# --------------------------------
test_oof = np.zeros(len(test))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for tr_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    model = XGBClassifier(**best_params)
    model.fit(X_tr, y_tr)

    # Fold modeli ile test tahmini
    test_oof += model.predict_proba(test.drop("id", axis=1))[:,1] / skf.n_splits

# Kaydet
pd.Series(test_oof, name="oof_pred_xgb_test").to_csv("oof_pred_xgb_test.csv", index=False)
print("Test OOF kaydedildi: oof_pred_xgb_test.csv")
