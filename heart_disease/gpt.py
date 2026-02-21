import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
X = train.drop("id", axis=1)
X_test = test.drop("id", axis=1)
y = X.pop("Heart Disease").map({"Presence": 1, "Absence": 0})


def add_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {
        "Number of vessels fluro",
        "Max HR",
        "Sex",
        "ST depression",
        "Slope of ST",
        "Cholesterol",
        "Exercise angina",
        "BP",
        "Age",
        "Chest pain type",
    }
    if required.issubset(df.columns):
        df["f1"] = (df["Number of vessels fluro"] == np.int64(3)) & (df["Max HR"] > 103.75) & (df["Sex"] == 1)
        df["f2"] = (df["Number of vessels fluro"] == np.int64(0)) & (df["ST depression"] > 1.86) & (df["Sex"] == 1)
        df["f3"] = (df["Slope of ST"] == np.int64(3)) & (df["ST depression"] < 0.62) & (df["Sex"] == 1)
        df["f4"] = (df["Slope of ST"] == np.int64(3)) & (df["Cholesterol"] > 235.50) & (df["Exercise angina"] == 1)
        df["f5"] = (df["Number of vessels fluro"] == np.int64(0)) & (df["BP"] < 120.50) & (df["Sex"] == 1)
        df["f6"] = (df["Slope of ST"] == np.int64(2)) & (df["ST depression"] > 1.24) & (df["Sex"] == 1)
        df["f7"] = (df["Slope of ST"] == np.int64(3)) & (df["Age"] < 61.00) & (df["Sex"] == 1)
        df["f8"] = (df["Chest pain type"] == np.int64(2)) & (df["Age"] > 45.00) & (df["Sex"] == 1)
        df["f9"] = (df["Chest pain type"] == np.int64(1)) & (df["Age"] < 45.00) & (df["Exercise angina"] == 1)
        df["ischemia_score"] = (
            df["ST depression"] * 3
            + df["Exercise angina"] * 2
            + df["Number of vessels fluro"] * 2
            + (df["Thallium"] >= 6).astype(int) * 2
        )
        df["maxhr_minus_age"] = df["Max HR"] - df["Age"]
        df["age_x_sex"] = df["Age"] * df["Sex"]
        df["stdep_x_exang"] = df["ST depression"] * df["Exercise angina"]
    return df


X = add_rule_features(X)
X_test = add_rule_features(X_test)


def add_oof_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    oof_pairs = [
        ("oof_pred_lgbm.csv", "oof_pred_lgbm_test.csv"),
        ("oof_pred_xgb.csv", "oof_pred_xgb_test.csv"),
        ("oof_pred_ridge.csv", "oof_pred_ridge_test.csv"),
    ]
    for train_file, test_file in oof_pairs:
        if os.path.exists(train_file) and os.path.exists(test_file):
            tr = pd.read_csv(train_file)
            te = pd.read_csv(test_file)
            if len(tr) == len(train_df) and len(te) == len(test_df):
                col_name = os.path.splitext(train_file)[0]
                train_df[col_name] = tr.iloc[:, 0].values
                test_df[col_name] = te.iloc[:, 0].values
    meta_train_path = os.path.join("data", "oof_train.csv")
    meta_test_path = os.path.join("data", "oof_test.csv")
    if os.path.exists(meta_train_path) and os.path.exists(meta_test_path):
        meta_tr = pd.read_csv(meta_train_path)
        meta_te = pd.read_csv(meta_test_path)
        if len(meta_tr) == len(train_df) and len(meta_te) == len(test_df):
            for col in meta_tr.columns:
                train_df[f"oof_{col}"] = meta_tr[col].values
                test_df[f"oof_{col}"] = meta_te[col].values
    return train_df, test_df


X, X_test = add_oof_features(X, X_test)


def get_cat_cols(df: pd.DataFrame) -> list[str]:
    cat_cols: list[str] = []
    for c in df.columns:
        dtype = df[c].dtype
        if dtype == "object" or str(dtype).startswith("category"):
            cat_cols.append(c)
            continue
        if pd.api.types.is_bool_dtype(dtype):
            cat_cols.append(c)
            continue
        if pd.api.types.is_integer_dtype(dtype) and df[c].nunique() <= 10:
            cat_cols.append(c)
            continue
    return cat_cols


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    cat_cols = get_cat_cols(df)
    num_cols = [c for c in df.columns if c not in cat_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )


def try_import_catboost():
    try:
        from catboost import CatBoostClassifier  # type: ignore

        return CatBoostClassifier
    except Exception:
        return None


def try_import_xgboost():
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier
    except Exception:
        return None


def try_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        return LGBMClassifier
    except Exception:
        return None


def prepare_catboost_df(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].astype(str).fillna("NA")
    return out


def cv_auc_catboost(model, X_df: pd.DataFrame, y_series: pd.Series, cat_cols: list[str]) -> float:
    X_cb = prepare_catboost_df(X_df, cat_cols)
    cat_features = [X_cb.columns.get_loc(c) for c in cat_cols]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in skf.split(X_cb, y_series):
        X_tr, X_va = X_cb.iloc[tr_idx], X_cb.iloc[va_idx]
        y_tr, y_va = y_series.iloc[tr_idx], y_series.iloc[va_idx]
        model.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
        pred = model.predict_proba(X_va)[:, 1]
        scores.append(roc_auc_score(y_va, pred))
    return float(np.mean(scores))


warnings.filterwarnings("ignore", category=UserWarning)

CV_FOLDS = 3

preprocessor = build_preprocessor(X)
cat_cols = get_cat_cols(X)

models: list[tuple[str, object, bool]] = []

CatBoostClassifier = try_import_catboost()
if CatBoostClassifier is not None:
    models.append(
        (
            "catboost",
            CatBoostClassifier(
                iterations=800,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=False,
            ),
            True,
        )
    )

XGBClassifier = try_import_xgboost()
if XGBClassifier is not None:
    models.append(
        (
            "xgboost",
            XGBClassifier(
                n_estimators=700,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=1.0,
                eval_metric="auc",
                tree_method="hist",
                random_state=42,
            ),
            False,
        )
    )

LGBMClassifier = try_import_lightgbm()
if LGBMClassifier is not None:
    models.append(
        (
            "lightgbm",
            LGBMClassifier(
                n_estimators=700,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            ),
            False,
        )
    )

models.extend(
    [
        (
            "hist_gb",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=400,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            ),
            False,
        ),
        (
            "hist_gb_tuned",
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_depth=7,
                max_iter=700,
                l2_regularization=0.0,
                max_leaf_nodes=255,
                min_samples_leaf=3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            ),
            False,
        ),
        (
            "hist_gb_aggressive",
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_depth=9,
                max_iter=900,
                l2_regularization=0.0,
                max_leaf_nodes=255,
                min_samples_leaf=1,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=25,
                random_state=42,
            ),
            False,
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=5,
                min_samples_split=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
            False,
        ),
        (
            "stack_hgb_et",
            StackingClassifier(
                estimators=[
                    (
                        "hgb",
                        HistGradientBoostingClassifier(
                            learning_rate=0.04,
                            max_depth=7,
                            max_iter=600,
                            l2_regularization=0.0,
                            max_leaf_nodes=127,
                            min_samples_leaf=5,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=20,
                            random_state=42,
                        ),
                    ),
                    (
                        "et",
                        ExtraTreesClassifier(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_leaf=5,
                            min_samples_split=5,
                            max_features="sqrt",
                            n_jobs=-1,
                            random_state=42,
                        ),
                    ),
                ],
                final_estimator=LogisticRegression(max_iter=300),
                stack_method="predict_proba",
                cv=CV_FOLDS,
                n_jobs=None,
            ),
            False,
        ),
    ]
)

best_name = None
best_score = -1.0
best_model = None
best_is_catboost = False

for name, model, is_catboost in models:
    if is_catboost:
        score = cv_auc_catboost(model, X, y, cat_cols)
    else:
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        score = float(np.mean(cross_val_score(pipe, X, y, cv=CV_FOLDS, scoring="roc_auc")))
    print(f"{name} cv roc_auc: {score:.5f}")
    if score > best_score:
        best_score = score
        best_name = name
        best_model = model
        best_is_catboost = is_catboost

print(f"best model: {best_name} (roc_auc={best_score:.5f})")

if best_is_catboost:
    X_cb = prepare_catboost_df(X, cat_cols)
    X_test_cb = prepare_catboost_df(X_test, cat_cols)
    cat_features = [X_cb.columns.get_loc(c) for c in cat_cols]
    best_model.fit(X_cb, y, cat_features=cat_features, verbose=False)
    test_pred = best_model.predict_proba(X_test_cb)[:, 1]
else:
    final_pipe = Pipeline([("prep", preprocessor), ("model", best_model)])
    final_pipe.fit(X, y)
    test_pred = final_pipe.predict_proba(X_test)[:, 1]

print(f"test_pred shape: {test_pred.shape}")

submission = pd.DataFrame({"id": test["id"], "Heart Disease": test_pred})
submission.to_csv("submission_gpt.csv", index=False)
print("saved submission_gpt.csv")
