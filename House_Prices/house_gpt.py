import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score , mean_squared_error , root_mean_squared_error
import optuna

# -----------------------
# Load data
# -----------------------
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

y = np.log1p(train["SalePrice"])
X = train.drop(columns=["SalePrice"])
X_test = test.copy()

# -----------------------
# ORDINAL ENCODING
# -----------------------
qual_map = {
    "Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0
}

ordinal_cols = [
    "ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
    "KitchenQual","FireplaceQu","GarageQual","GarageCond",
    "PoolQC","Fence"
]

for col in ordinal_cols:
    for df in [X, X_test]:
        df[col] = df[col].map(qual_map).fillna(0)

# -----------------------
# Column split
# -----------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# -----------------------
# Pipelines
# -----------------------
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("power", PowerTransformer(method="yeo-johnson"))
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# --------------------
# Optuna
# ---------------------

X_train_split, X_valid, y_train_split, y_valid = train_test_split(
    train, y, test_size=0.2, random_state=42
)

X_train_split_proc = preprocessor.fit_transform(X_train_split)
X_valid_proc = preprocessor.transform(X_valid)

def objective(trial):
    params = {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "max_iter": trial.suggest_int("max_iter", 100, 5000),
    }

    model = ElasticNet(**params, random_state=42)
    model.fit(X_train_split_proc, y_train_split)
    preds = model.predict(X_valid_proc)

    rmse = root_mean_squared_error(y_valid, preds)
    return rmse  # minimize

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, timeout=600)

# -----------------------
# Model
# -----------------------
model = ElasticNet(**study.best_trial.params)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# -----------------------
# Fit & Predict
# -----------------------
pipe.fit(X, y)

pred_log = pipe.predict(X_test)
pred = np.expm1(pred_log)

# -----------------------
# Submission
# -----------------------
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": pred
})

submission.to_csv("ordinal_skew_fixed.csv", index=False)
