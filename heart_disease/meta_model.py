import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMClassifier, early_stopping
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
    params = {
        "boosting_type": "gbdt",    
        "force_row_wise": True,
        # model params
        "learning_rate": trial.suggest_float("learning_rate", 0.01 , 0.9, log=True),
        "num_leaves": trial.suggest_int("num_leaves" ,10, 512),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "n_estimators":trial.suggest_int("n_estimators", 100 ,10000 ),
        "random_state": 42,
        "n_jobs": -1,
        'verbosity': -1
    }

    meta_model = LGBMClassifier(**params)
    meta_model.fit(X_train_meta, y_train_meta ,
                   eval_set=[(X_valid_meta, y_valid_meta)],
                   eval_metric="roc_auc",
                   callbacks=[early_stopping(200)])
    y_pred = meta_model.predict_proba(X_valid_meta)[:, 1]
    score = roc_auc_score(y_valid_meta, y_pred)
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best meta AUC:", study.best_value)
print("Best params:", study.best_params)

# En iyi parametrelerle tüm train OOF üzerinde meta modeli fit et
best_params = study.best_params
best_params.update({"random_state":42, "n_jobs":-1})

meta_model = LGBMClassifier(**best_params)
meta_model.fit(X_meta, y)

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
result.to_csv("meta_result_lgbm.csv", index=False)
print("Meta model LGBM ile sonuç kaydedildi: meta_result_lgbm.csv")
