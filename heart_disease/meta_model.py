import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, root_mean_squared_error

# Train OOF'ları yükle
ridge_oof = pd.read_csv("oof_pred_ridge.csv")
lgbm_oof = pd.read_csv("oof_pred_lgbm.csv")
xgb_oof  = pd.read_csv("oof_pred_xgb.csv")

y = pd.read_csv("./data/train.csv")["Heart Disease"].map({"Presence":1, "Absence":0})

X_meta = pd.DataFrame({
    "ridge": ridge_oof.iloc[:, 0],
    "lgbm": lgbm_oof.iloc[:, 0],
    #"xgb": xgb_oof.iloc[:, 0]
})

print("oof corr:",X_meta.corr())

# Holdout set oluştur
X_train_meta, X_valid_meta, y_train_meta, y_valid_meta = train_test_split(
    X_meta, y, test_size=0.075, stratify=y, random_state=42
)


def objective(trial):
    C = trial.suggest_float("C", 1e-3, 100.0, log=True)
    model = LogisticRegression(
    penalty="l2",      # ridge
    C=C,             # CV ile tune edilebilir
    solver="lbfgs",
    max_iter=1000
    )

    model.fit(X_train_meta , y_train_meta)
    probas = model.predict_proba(X_valid_meta)[:,1]

    return roc_auc_score(y_valid_meta, probas)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best AUC:", study.best_value)
print("Best params:", study.best_params)


best_C = study.best_params["C"]


meta_model = LogisticRegression(
    penalty="l2",      # ridge
    C=best_C,             # CV ile tune edilebilir
    solver="lbfgs",
    max_iter=1000
)
meta_model.fit(X_train_meta, y_train_meta)

# Test OOF'ları ile final tahmin
ridge_test_oof = pd.read_csv("oof_pred_ridge_test.csv")
lgbm_test_oof  = pd.read_csv("oof_pred_lgbm_test.csv")
xgb_test_oof   = pd.read_csv("oof_pred_xgb_test.csv")

X_meta_test = pd.DataFrame({
    "ridge": ridge_test_oof.iloc[:, 0],
    "lgbm": lgbm_test_oof.iloc[:, 0],
    #"xgb": xgb_test_oof.iloc[:, 0]
})

y_meta_test_pred = meta_model.predict_proba(X_meta_test)[:,1]

result = pd.DataFrame({
    "id": pd.read_csv("./data/test.csv")["id"],
    "Heart Disease": y_meta_test_pred
})
result.to_csv("meta_result_ridge.csv", index=False)
print("Meta model Ridge ile sonuç kaydedildi: meta_result_ridge.csv")
