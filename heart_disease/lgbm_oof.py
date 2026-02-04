import optuna
import pandas as pd
import numpy as np

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})


from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
        'max_depth': trial.suggest_int('max_depth', 3, 32),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'class_weight': None,  # dilersen 'balanced' da ekleyebilirsin
        'random_state': 42,
        'verbosity': -1
    }
    
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr ,
                  eval_set=[(X_val, y_val)],
                  eval_metric="roc_auc",
                  callbacks=[early_stopping(100),log_evaluation(0)])
        
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
    
    score = roc_auc_score(y, oof)
    return score


study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50)

print("Best ROC AUC:", study.best_value)
print("Best params:", study.best_params)


best_params = study.best_params  # Optuna’dan


oof_pred_lgbm = np.zeros(len(X))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    model = LGBMClassifier(**best_params)
    model.fit(X_tr, y_tr)
    
    oof_pred_lgbm[val_idx] = model.predict_proba(X_val)[:, 1]
    
    fold_auc = roc_auc_score(y_val, oof_pred_lgbm[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

print("OOF AUC:", roc_auc_score(y, oof_pred_lgbm))

pd.Series(oof_pred_lgbm, name="oof_pred_lgbm").to_csv("oof_pred_lgbm.csv", index=False)

# --------------------------------
# Test OOF üret
# --------------------------------
test_oof = np.zeros(len(test))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for tr_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    model = LGBMClassifier(**best_params)
    model.fit(X_tr, y_tr)

    # Fold modeli ile test tahmini
    test_oof += model.predict_proba(test.drop("id", axis=1))[:,1] / skf.n_splits

# Kaydet
pd.Series(test_oof, name="oof_pred_lgbm_test").to_csv("oof_pred_lgbm_test.csv", index=False)
print("Test OOF kaydedildi: oof_pred_lgbm_test.csv")