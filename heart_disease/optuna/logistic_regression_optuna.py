import optuna
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})


numeric_cols = [
    "ST depression",
    "Age",  "Cholesterol",
    "Max HR", "BP","Slope of ST"
]

categorical_cols = [
    "Thallium", "Chest pain type"
]

ordinal_cols = ["EKG results","Number of vessels fluro"]

binary_cols = [
    "Exercise angina","Sex", "FBS over 120" 
]


preprocess_v1 = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore") , categorical_cols+ ordinal_cols),
    #(TargetEncoder(cols=nominal_cols, smoothing=5) , nominal_cols ),
    (StandardScaler() , numeric_cols  ),
    remainder="passthrough"

)

preprocess_v2 = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore") , categorical_cols),
    #(TargetEncoder(cols=nominal_cols, smoothing=5) , nominal_cols ),
    (StandardScaler() , numeric_cols + ordinal_cols ),
    remainder="passthrough"

)

X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075, shuffle=True ,stratify=y)



logreg_ridge_v1 = LogisticRegression(
    penalty="l2",        # Ridge
    C=0.009125641392882999,               # Regularization gücü (küçük C = daha güçlü)
    solver="lbfgs",      # default ve stabil
    max_iter=1000,
    n_jobs=-1
)

logreg_ridge_v2 = LogisticRegression(
    penalty="l2",        # Ridge
    C=0.24425776679024996,               # Regularization gücü (küçük C = daha güçlü)
    solver="lbfgs",      # default ve stabil
    max_iter=1000,
    n_jobs=-1
)
#---
logreg_lasso_liblinear_v1 = LogisticRegression(
    penalty="l1",        # Lasso
    C=24.273648109254104,
    solver="liblinear", # veya "saga"
    max_iter=5000
)

logreg_lasso_liblinear_v2 = LogisticRegression(
    penalty="l1",        # Lasso
    C=0.380498218569746,
    solver="liblinear", # veya "saga"
    max_iter=5000
)
#---

logreg_lasso_saga_v2 = LogisticRegression(
    penalty="l1",        # Lasso
    C=0.004458034011942508,
    solver="saga", # veya "saga"
    max_iter=500
)

logreg_lasso_saga_v1 = LogisticRegression(
    penalty="l1",        # Lasso
    C=8.552003705570856,
    solver="saga", # veya "saga"
    max_iter=100
)
#-----
logreg_elastic_v1 = LogisticRegression(
    penalty="elasticnet",
    C=45.13119106050762,
    l1_ratio=0.5,       # 0 → Ridge, 1 → Lasso
    solver="saga",      # şart
    max_iter=100,
    n_jobs=-1
)

logreg_elastic_v2 = LogisticRegression(
    penalty="elasticnet",
    C=6.515177362156471,
    l1_ratio=0.5,       # 0 → Ridge, 1 → Lasso
    solver="saga",      # şart
    max_iter=100,
    n_jobs=-1
)

def objective(trial):

    C = trial.suggest_float("C", 1e-4, 50, log=True)

    model = LogisticRegression(
    penalty="elasticnet",
    C=C,
    l1_ratio=0.5,       # 0 → Ridge, 1 → Lasso
    solver="saga",      # şart
    max_iter=100,
    n_jobs=-1
    )


    X_train_proc = preprocess_v2.fit_transform(X_train)
    X_valid_proc = preprocess_v2.transform(X_valid)

    model.fit(X_train_proc, y_train)
    preds = model.predict_proba(X_valid_proc)[:, 1]

    return roc_auc_score(y_valid, preds)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=75)

print("Best ROC AUC:", study.best_value)
print("Best params:", study.best_params)