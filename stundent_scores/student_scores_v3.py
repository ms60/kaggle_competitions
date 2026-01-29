"""
Refactored and corrected version of your pipeline with NaN-safe Ridge handling.
Key fixes included:
 - consistent categorical lists between training, ridge OOF and test prediction
 - boolean -> int conversions
 - safeguards for column order when transforming test set
 - using model.best_iteration_ at predict time
 - rmse calculation compatible with older sklearn versions
 - NaN filling before Ridge and encoder usage
 - clearer comments and assertions

Save this file and run in the same folder as your ./data/train.csv and ./data/test.csv
"""

import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from category_encoders import TargetEncoder

# ---------- Helper functions ----------

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

# ---------- Read data ----------
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# ---------- Consistent feature lists (important) ----------
cat_nominal_ridge = ["gender", "course", "study_method"]  # for Ridge OOF
cat_ordinal = ["internet_access", "sleep_quality", "facility_rating", "exam_difficulty"]
num_basic = ["age", "study_hours", "class_attendance", "sleep_hours"]

# ---------- Basic mapping (ordinal) ----------
ord_map = {
    "internet_access": {"yes": 1, "no": 0},
    "sleep_quality": {"poor": 0, "average": 1, "good": 2},
    "facility_rating": {"low": 0, "medium": 1, "high": 2},
    "exam_difficulty": {"easy": 0, "moderate": 1, "hard": 2},
}

for col, m in ord_map.items():
    if col in train.columns:
        train[col] = train[col].map(m)
    if col in test.columns:
        test[col] = test[col].map(m)

# ---------- Feature engineering ----------

def add_features(df):
    df = df.copy()
    df["study_efficiency"] = df["study_hours"] * (df["class_attendance"] / 100.0)
    df["sleep_efficiency"] = df["sleep_hours"] * df["sleep_quality"]
    df["student_discipline_score"] = (
        0.4 * df["study_hours"] + 0.3 * df["class_attendance"] + 0.3 * df["sleep_efficiency"]
    )
    df["facility_study_interaction"] = df["study_hours"] * df["facility_rating"]
    df["low_attendance_flag"] = (df["class_attendance"] < 75.0).astype(int)
    df["sleep_deprivation_flag"] = ((df["sleep_hours"] < 6) & (df["study_hours"] > 5)).astype(int)
    df["study_hours_squared"] = df["study_hours"] ** 2
    df["study_hours_sqrt"] = np.sqrt(df["study_hours"])
    df["study_hours_log"] = np.log1p(df["study_hours"])
    df["study_hours_times_attendance"] = df["study_hours"] * df["class_attendance"]
    df["study_hours_times_sleep"] = df["study_hours"] * df["sleep_hours"]
    df["high_study_flag"] = (df["study_hours"] >= 7).astype(int)
    return df

train = add_features(train)
test = add_features(test)

X = train.drop(["id", "exam_score"], axis=1).copy()
y = train["exam_score"].copy()

# ---------- Ridge OOF generation (NaN-safe) ----------
X_oof = X.copy()
X_oof["ridge_oof"] = np.nan

kf = KFold(n_splits=5, shuffle=True, random_state=42)

ridge_pipe = make_pipeline(
    TargetEncoder(cols=cat_nominal_ridge, smoothing=5),
    Ridge(alpha=1.0)
)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_oof)):
    X_tr, X_val = X_oof.iloc[tr_idx], X_oof.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    X_tr = X_tr.fillna(-999)
    X_val = X_val.fillna(-999)

    ridge_pipe.fit(X_tr, y_tr)
    preds = ridge_pipe.predict(X_val)
    X_oof.iloc[val_idx, X_oof.columns.get_loc("ridge_oof")] = preds

assert X_oof["ridge_oof"].isnull().sum() == 0, "OOF column contains NaNs"

X_filtered = X_oof.copy()

# ---------- Ridge full fit for test prediction ----------
ridge_pipe_full = make_pipeline(
    TargetEncoder(cols=cat_nominal_ridge, smoothing=5),
    Ridge(alpha=1.0)
)

ridge_pipe_full.fit(X.fillna(-999), y)
test["ridge_oof"] = ridge_pipe_full.predict(test[X.columns].fillna(-999))

# ---------- Final preprocess ----------
cat_nominal_final = ["study_method", "gender", "course"]
cat_ordinal_final = ["sleep_quality", "facility_rating", "exam_difficulty"]

num_cols_final = [
    "ridge_oof", "age", "study_hours", "class_attendance", "sleep_hours",
    "study_efficiency", "sleep_efficiency", "student_discipline_score", "facility_study_interaction",
    "low_attendance_flag", "sleep_deprivation_flag", "study_hours_squared", "study_hours_sqrt",
    "study_hours_log", "study_hours_times_attendance", "study_hours_times_sleep", "high_study_flag"
]

preprocess_filtered = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_nominal_final),
    (StandardScaler(), num_cols_final),
    (StandardScaler(), cat_ordinal_final),
    remainder="drop",
)

X_train, X_valid, y_train, y_valid = train_test_split(X_filtered, y, test_size=0.075, random_state=42)

preprocess_filtered.fit(X_train)
X_train_proc = preprocess_filtered.transform(X_train)
X_valid_proc = preprocess_filtered.transform(X_valid)
X_test_proc = preprocess_filtered.transform(test[X_train.columns].copy())

# ---------- Model ----------
test_params = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.04652296188819206,
    'num_leaves': 216,
    'max_depth': 5,
    'min_child_samples': 230,
    'subsample': 0.9933406757691426,
    'colsample_bytree': 0.26465493457945716,
    'reg_alpha': 0.003176408880973185,
    'reg_lambda': 0.10726491808899498,
    'n_estimators': 7085,
    'random_state': 42,
    'n_jobs': -1,
}

model = LGBMRegressor(**test_params)

model.fit(
    X_train_proc, y_train,
    eval_set=[(X_valid_proc, y_valid)],
    eval_metric="rmse",
    callbacks=[early_stopping(200), log_evaluation(0)],
)

best_iter = getattr(model, "best_iteration_", None)
if best_iter is not None and best_iter > 0:
    preds_test = model.predict(X_test_proc, num_iteration=best_iter)
else:
    preds_test = model.predict(X_test_proc)

result = pd.DataFrame({"id": test["id"], "exam_score": preds_test})
result.to_csv("result.csv", index=False)

print("Finished successfully. result.csv saved.")