# fixed_pipeline.py — cleaned & improved version of your script
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# correct import for TargetEncoder
from category_encoders import TargetEncoder

from lightgbm import LGBMRegressor
import optuna

# ---- data ----
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

sortedColumns = train.drop("Id", axis=1).columns.tolist()
sortedColumns.sort()

X = train[sortedColumns].copy()
y = X.pop("SalePrice")
y = np.log1p(y)  # log target

X_test = test.drop("Id", axis=1).copy()

# convert year-ish features to age (relative)
X["YearBuilt"] = X["YearBuilt"] - X["YearBuilt"].min()
X["YearRemodAdd"] = X["YearRemodAdd"] - X["YearRemodAdd"].min()
X["GarageYrBlt"] = X["GarageYrBlt"] - X["GarageYrBlt"].min()

# ---- feature lists (kept your lists) ----
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# drop columns you intentionally excluded
numeric_features.remove("MoSold")
numeric_features.remove("YrSold")
categorical_features += ["MoSold", "YrSold"]

zeroFlagList = ["2ndFlrSF", "3SsnPorch", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                "EnclosedPorch", "GarageArea", "LowQualFinSF", "MSSubClass",
                "MasVnrArea", "MiscVal", "OpenPorchSF", "PoolArea", "ScreenPorch",
                "WoodDeckSF", "YearRemodAdd"]

log1pList = ["1stFlrSF","2ndFlrSF","3SsnPorch","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF",
             "EnclosedPorch","GarageArea","GarageYrBlt","GrLivArea","LotArea",
             "LotFrontage","MasVnrArea","MiscVal","OpenPorchSF","PoolArea",
             "ScreenPorch","TotalBsmtSF","WoodDeckSF","YearBuilt","YearRemodAdd"]

targetEncodingList = ["BedroomAbvGr","Fireplaces","FullBath","GarageCars","HalfBath",
                      "KitchenAbvGr","LowQualFinSF","MSSubClass","OverallCond",
                      "OverallQual","TotRmsAbvGrd"]

ohe_features = ["Alley","BldgType","CentralAir","Condition1","Condition2","Foundation",
                "Heating","LotConfig","MSZoning","MasVnrType","Street"]

te_features = ["Exterior1st","Exterior2nd","MiscFeature","Neighborhood",
               "RoofMatl","RoofStyle","SaleType","Utilities","MoSold","YrSold"]

ordinal_features = [col for col in categorical_features if col not in ohe_features + te_features]

# ---- simple engineered flags ----
X_numeric = X[numeric_features].copy()
for col in zeroFlagList:
    X_numeric[f"{col}_flag"] = (X_numeric[col] == 0).astype(int)

# apply same flags to test later (do it also below)
X_test_numeric = X_test[numeric_features].copy()
for col in zeroFlagList:
    X_test_numeric[f"{col}_flag"] = (X_test_numeric[col] == 0).astype(int)

# ---- categorical pre-cleaning (kept your mappings) ----
X_categorical = X[categorical_features].copy()
# ... (kept your mapping code here) ...
# For brevity: please reuse your mapping code block exactly as in the original script
# making sure to use .loc[:, col] = ... if pandas raises SettingWithCopyWarning

# ---- pipelines ----
# Target encoder pipeline: target encoder then scaler
te_pipeline = make_pipeline(
    TargetEncoder(smoothing=0.0),  # smoothing param depends on category_encoders version; you used "auto"
    StandardScaler()
)

# Preprocess overall
preprocess_total = make_column_transformer(
    # imputers
    (SimpleImputer(strategy='median'), [
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea',
        'GarageYrBlt', 'LotFrontage', 'MasVnrArea', 'TotalBsmtSF'
    ]),
    (SimpleImputer(strategy='most_frequent'), [
        'GarageCars', 'BsmtFullBath', 'BsmtHalfBath'
    ]),
    # log transform (FunctionTransformer). use validate=False to accept DataFrame
    (FunctionTransformer(lambda arr: np.log1p(arr), validate=False), log1pList),
    # standard scale numeric + ordinal
    (StandardScaler(), numeric_features + ordinal_features),
    # one-hot encode with unseen handling
    (OneHotEncoder(handle_unknown='ignore'), ohe_features),
    # target encode selected features
    (te_pipeline, te_features + targetEncodingList),
    remainder="passthrough"
)

# ---- train/val split for Optuna with early stopping ----
X_total = pd.concat([X_numeric, X_categorical], axis=1)  # as you had

X_train_total, X_val_total, y_train_total, y_val_total = train_test_split(
    X_total, y, test_size=0.2, random_state=42
)

# ---- objective with LightGBM early stopping ----
def objective_total(trial):
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": trial.suggest_int("n_estimators", 300, 3000),  # reduce upper bound
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0),
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1
    }

    model_total = LGBMRegressor(**params)
    pipeline_total = make_pipeline(preprocess_total, model_total)

    # Fit with early stopping using validation set
    pipeline_total.fit(
        X_train_total, y_train_total,
        lgbmregressor__eval_set=[(X_val_total, y_val_total)],
        lgbmregressor__early_stopping_rounds=50,
        lgbmregressor__verbose=False
    )

    preds_total = pipeline_total.predict(X_val_total)
    score_total = mean_squared_error(y_val_total, preds_total, squared=False)
    return score_total

study_total = optuna.create_study(direction="minimize", study_name="house_prices_total")
study_total.optimize(objective_total, n_trials=100)  # start with 100 trials

print("best params:", study_total.best_params)
print("best value (RMSE):", study_total.best_value)

# ---- final training with best params ----
best = study_total.best_params
final_model = LGBMRegressor(**best, random_state=42, n_jobs=-1)
final_pipeline = make_pipeline(preprocess_total, final_model)
final_pipeline.fit(X_total, y)

# Prepare test data (apply same preprocessing cleaning — reuse the same mapping code)
X_test_total = pd.concat([X_test_numeric, X_test[categorical_features]], axis=1)

test_preds_log = final_pipeline.predict(X_test_total)
test_preds = np.expm1(test_preds_log)

submission = pd.DataFrame({"Id": test["Id"], "SalePrice": test_preds})
submission.to_csv("result.csv", index=False)