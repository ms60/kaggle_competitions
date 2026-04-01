# ============================================
# 2-STAGE HYPOTHYROID PIPELINE (FULL)
# ============================================

from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np

from scipy.io import arff

from sklearn.metrics import (
    average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score , accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, train_test_split

pd.set_option('display.max_columns', None)

# ============================================
# LOAD DATA
# ============================================

data, meta = arff.loadarff("./data/dataset_57_hypothyroid.arff")
dataset = pd.DataFrame(data)

# decode byte → str
for col in dataset.select_dtypes([object]):
    dataset[col] = dataset[col].str.decode("utf-8")

# SAVE ORIGINAL LABEL
dataset["Class_original"] = dataset["Class"]

# ============================================
# FEATURE ENGINEERING
# ============================================

binary_cols = [
    "on_thyroxine","query_on_thyroxine","on_antithyroid_medication","sick",
    "pregnant","thyroid_surgery","I131_treatment","query_hypothyroid",
    "query_hyperthyroid","lithium","goitre","tumor","hypopituitary",
    "psych","TSH_measured","T3_measured","TT4_measured","T4U_measured",
    "FTI_measured","TBG_measured"
]

for col in binary_cols:
    dataset[col] = dataset[col].map({"f": 0, "t": 1})

dataset["sex"] = dataset["sex"].map({"F": 0, "M": 1})

dataset = dataset.drop("TBG", axis=1)
dataset["age"] = dataset["age"].fillna(dataset["age"].median())

# Stage 1 target (binary)
dataset["Class"] = dataset["Class"].apply(lambda x: 0 if x == "negative" else 1)

# ============================================
# TRAIN TEST SPLIT
# ============================================

X = dataset.drop(columns=["Class", "Class_original"])
y = dataset["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================
# OUTLIER CLIP (train-based)
# ============================================

num_cols = ["age","TSH","T3","TT4","T4U","FTI"]

for col in num_cols:
    lower = X_train[col].quantile(0.01)
    upper = X_train[col].quantile(0.99)
    X_train[col] = X_train[col].clip(lower, upper)
    X_test[col] = X_test[col].clip(lower, upper)

# ============================================
# MISSING + FLAGS
# ============================================

for col in ["sex","TSH","T3","TT4","T4U","FTI"]:
    X_train[f"{col}_isnull"] = X_train[col].isnull().astype(int)
    X_test[f"{col}_isnull"] = X_test[col].isnull().astype(int)

X_train["T4U_FTI_isnull"] = X_train["T4U"].isnull() & X_train["FTI"].isnull()
X_test["T4U_FTI_isnull"] = X_test["T4U"].isnull() & X_test["FTI"].isnull()

# fill values
fill_values = {
    "sex": X_train["sex"].mode()[0],
    "TSH": X_train["TSH"].median(),
    "T3": X_train["T3"].median(),
    "TT4": X_train["TT4"].median(),
    "T4U": X_train["T4U"].median(),
    "FTI": X_train["FTI"].median(),
}

for col, val in fill_values.items():
    X_train[col] = X_train[col].fillna(val)
    X_test[col] = X_test[col].fillna(val)

# category
X_train["referral_source"] = X_train["referral_source"].astype("category")
X_test["referral_source"] = X_test["referral_source"].astype("category")



# ============================================
# STAGE 1 MODEL
# ============================================

model1 = LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# CV OOF
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(X_train))

for tr_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    model1.fit(X_tr, y_tr)
    oof_proba[val_idx] = model1.predict_proba(X_val)[:,1]

print("Stage1 OOF ROC-AUC:", roc_auc_score(y_train, oof_proba))

# FULL TRAIN
model1.fit(X_train, y_train)

stage1_test_pred = model1.predict(X_test)

# ============================================
# STAGE 2 MODEL
# ============================================

# only positive from train
mask = (y_train == 1)
X_stage2 = X_train[mask].copy()

y_stage2_full = dataset.loc[X_stage2.index, "Class_original"]

# drop secondary
mask2 = y_stage2_full != "secondary_hypothyroid"
X_stage2 = X_stage2[mask2]
y_stage2_full = y_stage2_full[mask2]

# binary mapping
y_stage2 = y_stage2_full.map({
    "primary_hypothyroid": 0,
    "compensated_hypothyroid": 1
})

model2 = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model2.fit(X_stage2, y_stage2)

# ============================================
# FINAL PREDICTION (CHAINING)
# ============================================

final_pred = np.zeros(len(X_test))

# negative
final_pred[stage1_test_pred == 0] = 0

# positive → stage2
pos_idx = np.where(stage1_test_pred == 1)[0]

if len(pos_idx) > 0:
    X_test_pos = X_test.iloc[pos_idx]
    stage2_pred = model2.predict(X_test_pos)
    final_pred[pos_idx] = stage2_pred + 1

# ============================================
# TRUE LABEL (3 CLASS)
# ============================================

y_test_full = dataset.loc[X_test.index, "Class_original"]

y_test_full = y_test_full.map({
    "negative": 0,
    "primary_hypothyroid": 1,
    "compensated_hypothyroid": 2
})

# ============================================
# FINAL METRICS
# ============================================

print("\nFINAL PIPELINE METRICS")

print({
    "accuracy": accuracy_score(y_test_full, final_pred),
    "precision_macro": precision_score(y_test_full, final_pred, average="macro"),
    "recall_macro": recall_score(y_test_full, final_pred, average="macro"),
    "f1_macro": f1_score(y_test_full, final_pred, average="macro"),
})

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test_full, final_pred))

print(X_train.index.intersection(X_test.index))

from sklearn.metrics.pairwise import euclidean_distances

dist = euclidean_distances(X_train[num_cols], X_test[num_cols])
print(np.min(dist))