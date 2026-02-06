# greedy_boolean_builder.py

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from itertools import combinations, product
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# =====================================================
# FEATURE GENERATORS (WITH EXPRESSIONS)
# =====================================================

def generate_gt_lt_num(df, num_cols, step_counts):
    feats = {}
    for col, step in zip(num_cols, step_counts):
        mn, mx = df[col].min(), df[col].max()
        step_val = (mx - mn) / step

        for i in range(1, step):
            lt = mn + step_val * i
            gt = mx - step_val * i

            feats[f"{col}_lt_{lt:.2f}"] = (
                df[col] < lt,
                f'(X["{col}"] < {lt:.2f})'
            )
            feats[f"{col}_gt_{gt:.2f}"] = (
                df[col] > gt,
                f'(X["{col}"] > {gt:.2f})'
            )
    return feats


def generate_category_equality(df, cat_cols):
    feats = {}
    for col in cat_cols:
        for v in df[col].dropna().unique():
            feats[f"{col}_eq_{v}"] = (
                df[col] == v,
                f'(X["{col}"] == {repr(v)})'
            )
    return feats


def generate_cross_combinations_same(feats, ops=("and",)):
    out = {}
    keys = list(feats.keys())
    for k1, k2 in combinations(keys, 2):
        v1, e1 = feats[k1]
        v2, e2 = feats[k2]

        if "and" in ops:
            out[f"{k1}_AND_{k2}"] = (
                v1 & v2,
                f"{e1} & {e2}"
            )
        if "or" in ops:
            out[f"{k1}_OR_{k2}"] = (
                v1 | v2,
                f"{e1} | {e2}"
            )
    return out


def generate_cross_combinations_different(f1, f2, ops=("and",)):
    out = {}
    for (k1, (v1, e1)), (k2, (v2, e2)) in product(f1.items(), f2.items()):
        if "and" in ops:
            out[f"{k1}_AND_{k2}"] = (
                v1 & v2,
                f"{e1} & {e2}"
            )
        if "or" in ops:
            out[f"{k1}_OR_{k2}"] = (
                v1 | v2,
                f"{e1} | {e2}"
            )
    return out


# =====================================================
# GREEDY AUC SELECTION
# =====================================================

def greedy_auc_feature_addition(
    X,
    y,
    candidate_features,
    model,
    test_size=0.075,
    random_state=42
):
    X_work = X.copy()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_work,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    model.fit(X_tr, y_tr)
    base_auc = roc_auc_score(
        y_val,
        model.predict_proba(X_val)[:, 1]
    )

    print(f"Initial AUC: {base_auc:.5f}")

    accepted = {}

    for name, (col, expr) in tqdm(
        candidate_features.items(),
        total=len(candidate_features),
        desc="Evaluating feature combinations"
    ):
        X_work[name] = col.astype("int8")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_work,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        model.fit(X_tr, y_tr)
        auc = roc_auc_score(
            y_val,
            model.predict_proba(X_val)[:, 1]
        )

        if auc > base_auc:
            base_auc = auc
            accepted[name] = (col, expr)
            tqdm.write(f"[KEEP] {expr} → AUC {auc:.5f}")
        else:
            X_work.drop(columns=[name], inplace=True)

    print(f"\nSelected {len(accepted)} features")
    print(f"Final AUC: {base_auc:.5f}")

    return accepted, X_work


# =====================================================
# RUN
# =====================================================

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id", axis=1)
y = X.pop("Heart Disease")
y = y.map({"Presence": 1, "Absence": 0})

X["f1"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Sex"] == 1)
X["f2"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f3"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f4"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)

X["f5"] =   (X["Sex"] == 1) & (X["Chest pain type"] == 3) & (X["EKG results"] == 2)

X["f6"] = (X["Thallium"] == 3) & (X["Age"] < 53.00)


numeric_cols = [
    "Age", "BP", "Cholesterol",
    "Max HR", "ST depression",
    "Slope of ST", "Number of vessels fluro"
]

categorical_cols = [
    "Thallium", "Chest pain type", "EKG results"
]

binary_cols = [
    "Sex", "Exercise angina", "FBS over 120"
]


lgbm_params = {
    'n_estimators': 724,
    'max_depth': 2,
    'num_leaves': 153,
    'min_child_samples': 99,
    'learning_rate': 0.1387114580881059,
    'subsample': 0.37549286841241186,
    'colsample_bytree': 0.9077375200328026,
    'reg_alpha': 0.6578963730687483,
    'reg_lambda': 0.28960307157515247
}

model = LGBMClassifier(**lgbm_params, verbose=-1 , n_jobs=-1)


num_features = generate_gt_lt_num(
    X, numeric_cols, [6, 5, 5, 5, 5, 5, 3]
)

cat_features = generate_category_equality(
    X, categorical_cols
)

base_booleans = {
    col: (
        X[col].astype(bool),
        f'(X["{col}"] == 1)'
    )
    for col in binary_cols
}

cat_cat_features = generate_cross_combinations_same(cat_features)
catcat_cat = generate_cross_combinations_different(cat_cat_features,cat_features)


num_num_features = generate_cross_combinations_same(num_features)
numnum_num = generate_cross_combinations_different(num_num_features,num_features)


total = generate_cross_combinations_different(
    cat_features,
    num_features
)

# grand_total = generate_cross_combinations_different(
    
#     base_booleans,
#     total,
#     ops=("and",)
# )


accepted, X_final = greedy_auc_feature_addition(
    X,
    y,
    numnum_num,
    model
)


# =====================================================
# OPTIONAL: DUMP ACCEPTED RULES
# =====================================================

print("\nAccepted rules (copy-paste ready):\n")
for _, (_, expr) in accepted.items():
    print(expr)
