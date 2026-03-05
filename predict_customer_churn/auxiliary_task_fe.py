from lightgbm import LGBMClassifier, LGBMRegressor
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

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

train["InternetQuality"] =   (train["OnlineSecurity"] + train["OnlineBackup"] + train["DeviceProtection"] + train["TechSupport"] + train["StreamingTV"] + train["StreamingMovies"] )
test["InternetQuality"] =  test["OnlineSecurity"] + test["OnlineBackup"] + test["DeviceProtection"] + test["TechSupport"] + test["StreamingTV"] + test["StreamingMovies"] 

train["TotalCharges_over_tenure"] = train["TotalCharges"] / train["tenure"]
test["TotalCharges_over_tenure"] = test["TotalCharges"] / test["tenure"]

train["Security"] = train["OnlineSecurity"] * train["DeviceProtection"] * train["TechSupport"]
test["Security"] = test["OnlineSecurity"] * test["DeviceProtection"] * test["TechSupport"]

train["ExtraFeatures"] = train["StreamingTV"] * train["StreamingMovies"]
test["ExtraFeatures"] = test["StreamingTV"] * test["StreamingMovies"]


train["Churn"] = train["Churn"].map({"No": 0, "Yes": 1})



X = train.drop(["id","Churn"],axis=1)
y = X.pop("TotalCharges")

X_test = test.drop(["id","TotalCharges"],axis=1)

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
ohe_features = ["Contract","PaymentMethod"]
for col in ohe_features:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")


X_train , X_valid , y_train , y_valid = train_test_split(X , y , test_size=0.075 , random_state=42 )

X_train_2 , X_valid_2 , y_train_2 , y_valid_2 = train_test_split(X , y , test_size=0.075 , random_state=60)


def objective(trial):

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
    
        "verbosity":-1,
    
        "force_row_wise": True,
    
        # model params
        "n_estimators":trial.suggest_int("n_estimators", 500 ,5000 ),
        "learning_rate": trial.suggest_float("learning_rate", 0.01 , 0.9, log=True),
        "num_leaves": trial.suggest_int("num_leaves" ,10, 512),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42
    }



    model = LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
    )

    y_pred = model.predict(X_valid)
    rmse = root_mean_squared_error(y_valid, y_pred)

    model_2 = LGBMRegressor(**params)
    model_2.fit(
        X_train_2,
        y_train_2,
        eval_set=[(X_valid_2, y_valid_2)],
        eval_metric="rmse",
    )

    y_pred_2 = model_2.predict(X_valid_2)
    rmse_2 = root_mean_squared_error(y_valid_2, y_pred_2)

    return (rmse + rmse_2) / 2

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)

# print(study.best_params)

#{'n_estimators': 3258, 'learning_rate': 0.07244553739566428, 'num_leaves': 298, 'max_depth': 7, 'min_child_samples': 261, 'subsample': 0.639873800688788, 'colsample_bytree': 0.8313859902463943, 'reg_alpha': 0.015832159846783444, 'reg_lambda': 0.08643257647830549}
best_params = {'n_estimators': 4906, 'learning_rate': 0.040971831282512576, 'num_leaves': 329, 'max_depth': 8, 'min_child_samples': 299, 'subsample': 0.62569502880929, 'colsample_bytree': 0.7544039280565649, 'reg_alpha': 6.59706550394261, 'reg_lambda': 0.038469688191257116}
best_params.update({
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "verbosity":-1,
    "force_row_wise": True,
    "random_state": 42
})

oof_preds = np.zeros(len(X), dtype=float) 
test_preds = np.zeros(len(X_test), dtype=float) 

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):

    print(f"Fold {fold+1}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = LGBMRegressor(**best_params)
    model.fit(X_train,y_train)

    oof_preds[valid_idx] = model.predict(X_valid)
    test_preds += model.predict(X_test) / kf.n_splits

X["TotalCharges_Predicted"] = oof_preds
X_test["TotalCharges_Predicted"] = test_preds

X["TotalCharges_residual"] = train["TotalCharges"] - X["TotalCharges_Predicted"]
X_test["TotalCharges_residual"] = test["TotalCharges"] - X_test["TotalCharges_Predicted"]

X["TotalCharges_residual"].to_csv("./stack/TotalCharges_residual.csv")
X_test["TotalCharges_residual"].to_csv("./stack/TotalCharges_residual_test.csv")