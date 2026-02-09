from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id", axis=1)
y = X.pop("Heart Disease")
y = y.map({"Presence": 1, "Absence": 0})

numeric_cols = [
    "ST depression",
    "Age",  "Cholesterol",
    "Max HR", "BP"
    
]

categorical_cols = [
    "EKG results","Thallium", "Chest pain type","Slope of ST","Number of vessels fluro"
]

binary_cols = [
    "Exercise angina","Sex", "FBS over 120"
]


# Tree based discritization

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

class DiscretizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_leaf_nodes=5,
        min_samples_leaf=50,
        encode=None  # None | "ordinal" | "onehot"
    ):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.encode = encode

    def fit(self, X, y):
        self.bins_ = {}
        self.encoders_ = {}

        for col in X.columns:
            tree = DecisionTreeClassifier(
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42
            )
            tree.fit(X[[col]], y)

            thresholds = tree.tree_.threshold
            splits = sorted(thresholds[thresholds != -2])
            bins = [-np.inf] + splits + [np.inf]

            self.bins_[col] = bins

            if self.encode == "onehot":
                cats = pd.cut(X[col], bins=bins).astype("category")
                ohe = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore"
                )
                ohe.fit(cats.to_frame())
                self.encoders_[col] = ohe

        return self

    def transform(self, X):
        X_out = []

        for col in X.columns:
            binned = pd.cut(X[col], bins=self.bins_[col])

            if self.encode == "ordinal":
                X_out.append(binned.cat.codes.to_frame(col + "_bin"))

            elif self.encode == "onehot":
                ohe = self.encoders_[col]
                arr = ohe.transform(binned.astype("category").to_frame())
                cols = [f"{col}_bin_{c}" for c in ohe.categories_[0]]
                X_out.append(pd.DataFrame(arr, columns=cols, index=X.index))

            else:
                X_out.append(binned.to_frame(col + "_bin"))

        return pd.concat(X_out, axis=1)
    

X["f1"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Sex"] == 1)
X["f2"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f3"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f4"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)

X["f5"] =   (X["Sex"] == 1) & (X["Chest pain type"] == 3) & (X["EKG results"] == 2)

X["f6"] = (X["Thallium"] == 3) & (X["Age"] < 53.00)



test["f1"] = (test["Age"] < 37.00) & (test["Thallium"] == 7) & (test["Sex"] == 1)
test["f2"] = (test["Age"] < 37.00) & (test["Thallium"] == 7) & (test["Exercise angina"] == 1)
test["f3"] = (test["Age"] > 69.00) & (test["Thallium"] == 7) & (test["Exercise angina"] == 1)
test["f4"] = (test["Age"] > 69.00) & (test["Thallium"] == 7) & (test["FBS over 120"] == 1)

test["f5"] =   (test["Sex"] == 1) & (test["Chest pain type"] == 3) & (test["EKG results"] == 2)

test["f6"] = (test["Thallium"] == 3) & (test["Age"] < 53.00)


preprocess = make_column_transformer(
    ( DiscretizerTransformer(encode="onehot") , numeric_cols ),
    ( OneHotEncoder() , categorical_cols ),
    remainder="passthrough"
)

# model = LogisticRegression(
#     C=1.0,
#     penalty="l2",
#     solver="lbfgs",
#     max_iter=2000
# )





def objective(trial):

    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "n_estimators": 300,   # SABİT
        "random_state": 42,
        "verbosity": -1,
    }

    model = LGBMClassifier(**params)

    pipe = make_pipeline(preprocess, model)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipe,
        X, y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    return scores.mean()

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=150)

# print("Best ROC AUC:", study.best_value)
# print("Best params:", study.best_params)

best_params  = {'learning_rate': 0.08122738603464233, 'num_leaves': 28, 'max_depth': 3, 'min_child_samples': 50, 'subsample': 0.9359982919468567, 'colsample_bytree': 0.9575328797160576, 'reg_alpha': 0.0016778300037891358, 'reg_lambda': 0.020763919127347433}
best_params.update({
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 300,
    "random_state": 42,
    "verbosity": -1
})

best_model = LGBMClassifier(**best_params)

pipe = make_pipeline(preprocess, best_model)
#pipe.fit(X,y)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(
    pipe,
    X, y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print(scores.mean())

# test_pred = pipe.predict_proba(test)[:, 1]


# result = pd.DataFrame({"id":test["id"] , "Heart Disease":test_pred})
# result.to_csv("logisticreg.csv",index=False)
