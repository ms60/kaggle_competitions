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

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best OOF AUC:", study.best_value)
print("Best params:", study.best_params)

best_params = study.best_params
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0
})

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
