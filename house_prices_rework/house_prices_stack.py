import optuna
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

sortedColumns = train.drop("Id",axis=1).columns.tolist()
sortedColumns.sort()


X = train[sortedColumns]
y = X.pop("SalePrice")
y = np.log1p(y)

sortedColumns.remove("SalePrice")
X_test = test.drop("Id",axis=1)
X_test = X_test[sortedColumns]

oof_numeric = pd.read_csv("./numeric_oof.csv")
oof_numeric_test = pd.read_csv("./numeric_oof_test.csv")

oof_categorical = pd.read_csv("./categorical_oof.csv")
oof_categorical_test = pd.read_csv("./categorical_oof_test.csv")

oof_total = pd.concat([oof_numeric,oof_categorical],axis=1)
oof_total_test = pd.concat([oof_numeric_test,oof_categorical_test],axis=1)

print(oof_total.head())
print(oof_total_test.head())

X_train, X_val, y_train, y_val = train_test_split(
    oof_total, y, test_size=0.2, random_state=42
)


def objective_elastic(trial):
    params = {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "max_iter": trial.suggest_int("max_iter", 100, 15000),
    }

    model_stack = ElasticNet(**params, random_state=42)


    model_stack.fit(X_train, y_train)

    preds_stack = model_stack.predict(X_val)

    score_stack = root_mean_squared_error(y_val, preds_stack)
    return score_stack



study_elastic = optuna.create_study(direction="minimize")
study_elastic.optimize(objective_elastic, n_trials=500)

best_params_elastic = study_elastic.best_params
print(best_params_elastic)

# def objective_total(trial):
#     params = {
#         "objective": "reg:squarederror",
#         "tree_method": "hist",  # veya "gpu_hist" (GPU varsa)
#         "eval_metric": "rmse",
#         "booster": trial.suggest_categorical("booster", ["gbtree"]), #, "dart"
#         #'early_stopping_rounds': 100,
        
#         # Öğrenme oranı ve derinlik
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.9, log=True),
#         "max_depth": trial.suggest_int("max_depth", 3, 32),
#         "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        
#         # Düzenlileştirme
#         "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),   # L2
#         "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),     # L1
        
#         # Alt örnekleme
#         "subsample": trial.suggest_float("subsample", 0.1, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        
#         # DART için özel dropout parametreleri (booster=dart olduğunda aktif)
#         "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.5),
#         "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.5),
        
#         # Ağaç sayısı
#         "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
#         "n_jobs": -1,
#         "random_state": 42,
        
#     }

#     model_stack = XGBRegressor(**params)
#     model_stack.fit(X_train, y_train)

#     preds_stack = model_stack.predict(X_val)

#     score_stack = root_mean_squared_error(y_val, preds_stack)
#     return score_stack


# study_total = optuna.create_study(
#     direction="minimize",
#     study_name="house_prices_total"
# )

# study_total.optimize(objective_total, n_trials=500)

# best_params = study_total.best_params
# print(best_params)

#{'booster': 'gbtree', 'learning_rate': 0.03491570018467506, 'max_depth': 3, 'min_child_weight': 5.479429768667506, 'lambda': 0.0037454652709622274, 'alpha': 0.8827039131877998, 'subsample': 0.9599756361584312, 'colsample_bytree': 0.7198054950111841, 'rate_drop': 0.3697630321025589, 'skip_drop': 0.29929030804225715, 'n_estimators': 2812}
#0.13069487075703515.