from category_encoders import TargetEncoder
import optuna
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder , StandardScaler

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})


nominal_cols = ["Sex","Chest pain type","Thallium"] 
num_cols = ["Age","BP","Cholesterol","Max HR","ST depression","Slope of ST",]
#te_cols = ["EKG results","Sex","Chest pain type","Thallium"]
ordinal_cols = ["EKG results","Number of vessels fluro"]
yes_or_no = ["FBS over 120","Exercise angina","Age_flag","BP_flag","Max_HR_flag","Cholesterol_flag"]

preprocess = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore") , nominal_cols),
    #(TargetEncoder(cols=nominal_cols, smoothing=5) , nominal_cols ),
    (StandardScaler() , num_cols + ordinal_cols ),
    remainder="passthrough"

)


from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def objective(trial):
    C = trial.suggest_float("C", 1e-3, 100.0, log=True)

    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", LogisticRegression(
                penalty="l2",
                C=C,
                solver="lbfgs",
                max_iter=1000,
                n_jobs=-1
            ))
        ])

        pipe.fit(X_tr, y_tr)
        oof[val_idx] = pipe.predict_proba(X_val)[:, 1]

    return roc_auc_score(y, oof)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best AUC:", study.best_value)
print("Best params:", study.best_params)


best_C = study.best_params["C"]

oof_pred_ridge = np.zeros(len(X))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(
            penalty="l2",
            C=best_C,
            solver="lbfgs",
            max_iter=1000,
            n_jobs=-1
        ))
    ])

    pipe.fit(X_tr, y_tr)
    oof_pred_ridge[val_idx] = pipe.predict_proba(X_val)[:, 1]

    fold_auc = roc_auc_score(y_val, oof_pred_ridge[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

print("OOF AUC:", roc_auc_score(y, oof_pred_ridge))

pd.Series(oof_pred_ridge, name="oof_pred_ridge").to_csv(
    "oof_pred_ridge.csv",
    index=False
)

# --------------------------------
# Test OOF üret
# --------------------------------
test_oof = np.zeros(len(test))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for tr_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(
            penalty="l2",
            C=best_C,
            solver="lbfgs",
            max_iter=1000,
            n_jobs=-1
        ))
    ])
    pipe.fit(X_tr, y_tr)

    # Fold modeli ile test tahmini
    test_oof += pipe.predict_proba(test.drop("id", axis=1))[:,1] / skf.n_splits

# Kaydet
pd.Series(test_oof, name="oof_pred_ridge_test").to_csv("oof_pred_ridge_test.csv", index=False)
print("Test OOF kaydedildi: oof_pred_ridge_test.csv")