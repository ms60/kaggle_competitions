from lightgbm import LGBMClassifier , early_stopping
import optuna
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler , TargetEncoder , LabelEncoder

from sklearn.feature_selection import mutual_info_classif
import shap






train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

train["gender"] = train["gender"].map({"Female": 0, "Male": 1})
test["gender"] = test["gender"].map({"Female": 0, "Male": 1})

train["Partner"] = train["Partner"].map({"No": 0, "Yes": 1})
test["Partner"] = test["Partner"].map({"No": 0, "Yes": 1})

train["Dependents"] = train["Dependents"].map({"No": 0, "Yes": 1})
test["Dependents"] = test["Dependents"].map({"No": 0, "Yes": 1})

train["PhoneService"] = train["PhoneService"].map({"No": 0, "Yes": 1})
test["PhoneService"] = test["PhoneService"].map({"No": 0, "Yes": 1})


train["MultipleLines"] = train["MultipleLines"].map({"No": 0, "Yes": 1, "No phone service": 0})
test["MultipleLines"] = test["MultipleLines"].map({"No": 0, "Yes": 1, "No phone service": 0})

train["InternetService"] = train["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2})
test["InternetService"] = test["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2})

train["OnlineSecurity"] = train["OnlineSecurity"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["OnlineSecurity"] = test["OnlineSecurity"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["OnlineBackup"] = train["OnlineBackup"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["OnlineBackup"] = test["OnlineBackup"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["DeviceProtection"] = train["DeviceProtection"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["DeviceProtection"] = test["DeviceProtection"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["TechSupport"] = train["TechSupport"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["TechSupport"] = test["TechSupport"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["StreamingTV"] = train["StreamingTV"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["StreamingTV"] = test["StreamingTV"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["StreamingMovies"] = train["StreamingMovies"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["StreamingMovies"] = test["StreamingMovies"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["PaperlessBilling"] = train["PaperlessBilling"].map({"No": 0, "Yes": 1})
test["PaperlessBilling"] = test["PaperlessBilling"].map({"No": 0, "Yes": 1})

train["Churn"] = train["Churn"].map({"No": 0, "Yes": 1})



X = train.drop("id",axis=1)
y = X.pop("Churn")

#X_test = test.drop("id",axis=1)

for col in ["Contract","PaymentMethod"]:
    X[col] = X[col].astype("category")
    #X_test[col] = X_test[col].astype("category")


X_small, _, y_small, _ = train_test_split(
    X,
    y,
    train_size=120_000,
    stratify=y,
    shuffle=True,
    random_state=42
)

    # params = {
    #     "n_estimators": trial.suggest_int("n_estimators", 70_000, 120_000),
    #     "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
    #     "max_depth": trial.suggest_int("max_depth", 3, 5),
    #     "num_leaves": trial.suggest_int("num_leaves", 20, 200),
    #     "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    #     "max_bin":trial.suggest_int("max_bin" , 10_000,30_000) ,
    #     "random_state": 42,
    #     "n_jobs": -1,
    #     "verbosity":-1,
    # }

#-------------------


from optuna.integration import LightGBMPruningCallback

# =========================
# DATA (örnek)
# =========================
# X, y zaten tanımlı olmalı
# örnek:
# df = pd.read_csv("data.csv")
# X = df.drop("target", axis=1)
# y = df["target"]

# =========================
# CONFIG
# =========================
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
N_TRIALS = 50
RANDOM_STATE = 42

# =========================
# OBJECTIVE (INNER CV)
# =========================
def objective(trial, X, y):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 70_000, 120_000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 5),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "max_bin":trial.suggest_int("max_bin" , 10_000,30_000) ,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity":-1
    }

    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = []

    for train_idx, valid_idx in inner_cv.split(X, y):

        # ✅ pandas-safe indexing
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = LGBMClassifier(**params)



        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                early_stopping(50),
                #LightGBMPruningCallback(trial, "auc")
            ],
        )

        preds = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, preds)

        scores.append(score)

    return np.mean(scores)

# =========================
# OUTER CV
# =========================
# outer_cv = StratifiedKFold(
#     n_splits=N_OUTER_SPLITS,
#     shuffle=True,
#     random_state=RANDOM_STATE
# )

# outer_scores = []
# best_params_list = []

# for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_small, y_small)):

#     print(f"\n========== OUTER FOLD {fold+1} ==========")

#     X_train = X.iloc[train_idx]
#     X_test = X.iloc[test_idx]
#     y_train = y.iloc[train_idx]
#     y_test = y.iloc[test_idx]

#     # =========================
#     # OPTUNA
#     # =========================
#     study = optuna.create_study(direction="maximize")

#     study.optimize(
#         lambda trial: objective(trial, X_train, y_train),
#         n_trials=N_TRIALS,
#         show_progress_bar=True
#     )

#     best_params = study.best_params
#     best_params["random_state"] = RANDOM_STATE
#     best_params["n_jobs"] = -1

#     best_params_list.append(best_params)

#     # =========================
#     # FINAL MODEL (OUTER TEST)
#     # =========================
#     model = LGBMClassifier(**best_params)

#     model.fit(X_train, y_train)

#     preds = model.predict_proba(X_test)[:, 1]
#     score = roc_auc_score(y_test, preds)

#     outer_scores.append(score)

#     print(f"Fold ROC-AUC: {score:.5f}")
#     print(f"Best Params: {best_params}")

# # =========================
# # FINAL RESULTS
# # =========================
# print("\n==============================")
# print(f"Mean ROC-AUC: {np.mean(outer_scores):.5f}")
# print(f"Std ROC-AUC: {np.std(outer_scores):.5f}")

study = optuna.create_study(direction="maximize")

study.optimize(
    lambda trial: objective(trial, X, y),
    n_trials=N_TRIALS,
    #show_progress_bar=True
)

print( study.best_params )