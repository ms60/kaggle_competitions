import optuna
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold , StratifiedKFold




train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})


logreg_ridge = LogisticRegression(
    penalty="l2",        # Ridge
    C=1.0,               # Regularization gücü (küçük C = daha güçlü)
    solver="lbfgs",      # default ve stabil
    max_iter=1000,
    n_jobs=-1
)

logreg_lasso = LogisticRegression(
    penalty="l1",        # Lasso
    C=1.0,
    solver="liblinear", # veya "saga"
    max_iter=1000
)

logreg_elastic = LogisticRegression(
    penalty="elasticnet",
    C=1.0,
    l1_ratio=0.5,       # 0 → Ridge, 1 → Lasso
    solver="saga",      # şart
    max_iter=1000,
    n_jobs=-1
)


####### elastic

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_pred_elastic = np.zeros(len(X))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = LogisticRegression(
    penalty="elasticnet",
    C=1.0,
    l1_ratio=0.5,       # 0 → Ridge, 1 → Lasso
    solver="saga",      # şart
    max_iter=1000,
    n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    # probability → [:, 1] ÇOK ÖNEMLİ
    oof_pred_elastic[val_idx] = model.predict_proba(X_val)[:, 1]

    fold_auc = roc_auc_score(y_val, oof_pred_elastic[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

# Genel OOF skor
print("OOF AUC:", roc_auc_score(y, oof_pred_elastic))


####### ridge

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_pred_ridge = np.zeros(len(X))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = LogisticRegression(
    penalty="l2",        # Ridge
    C=1.0,               # Regularization gücü (küçük C = daha güçlü)
    solver="lbfgs",      # default ve stabil
    max_iter=1000,
    n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    # probability → [:, 1] ÇOK ÖNEMLİ
    oof_pred_ridge[val_idx] = model.predict_proba(X_val)[:, 1]

    fold_auc = roc_auc_score(y_val, oof_pred_ridge[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

# Genel OOF skor
print("OOF AUC:", roc_auc_score(y, oof_pred_ridge))


####### lasso

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_pred_lasso = np.zeros(len(X))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = LogisticRegression(
    penalty="l1",        # Lasso
    C=1.0,
    solver="liblinear", # veya "saga"
    max_iter=1000
    )
    model.fit(X_tr, y_tr)

    # probability → [:, 1] ÇOK ÖNEMLİ
    oof_pred_lasso[val_idx] = model.predict_proba(X_val)[:, 1]

    fold_auc = roc_auc_score(y_val, oof_pred_lasso[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

# Genel OOF skor
print("OOF AUC:", roc_auc_score(y, oof_pred_lasso))