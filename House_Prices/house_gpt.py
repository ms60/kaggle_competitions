import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet

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

# -----------------------
# Model
# -----------------------
model = ElasticNet(
    alpha=0.0005,
    l1_ratio=0.9,
    random_state=42
)

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
