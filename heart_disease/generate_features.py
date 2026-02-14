from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, cross_val_score


class IndicatorFeatureGenerator:
    def __init__(
        self,
        model,
        metric,
        cv=5,
        min_gain=1e-4,
        random_state=42
    ):
        """
        model  : sklearn / lgbm compatible classifier
        metric : string (e.g. 'roc_auc')
        """
        self.model = model
        self.metric = metric
        self.cv = cv
        self.min_gain = min_gain
        self.random_state = random_state

        self.kept_features = []
        self.feature_defs = {}
        self.base_score = None

    # --------------------------------------------------
    def baseline_score(self, X, y):
        cv = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        score = cross_val_score(
            self.model, X, y,
            scoring=self.metric,
            cv=cv,
            n_jobs=-1
        ).mean()
        self.base_score = score
        return score

    # --------------------------------------------------
    def generate_atomic_candidates(
        self,
        X,
        numeric_cols,
        categorical_cols,
        binary_cols,
        percentiles=(10, 25, 50, 75),
    ):
        """
        Atomic building blocks:
        - num > percentile
        - num < percentile
        - cat == value
        - binary == 1
        """
        candidates = {}

        # numeric thresholds
        for col in numeric_cols:
            qs = np.percentile(X[col].dropna(), percentiles)
            for q, p in zip(qs, percentiles):
                candidates[f"{col}_gt_p{p}"] = lambda df, c=col, v=q: (df[c] > v)
                candidates[f"{col}_lt_p{p}"] = lambda df, c=col, v=q: (df[c] < v)

        # categorical
        for col in categorical_cols:
            for val in X[col].dropna().unique():
                candidates[f"{col}_eq_{val}"] = (
                    lambda df, c=col, v=val: (df[c] == v)
                )

        # binary
        for col in binary_cols:
            candidates[f"{col}_is1"] = lambda df, c=col: (df[c] == 1)

        return candidates

    # --------------------------------------------------
    def evaluate_single_feature(self, X, y, feature_name, feature_fn):
        X_tmp = X.copy()
        X_tmp[feature_name] = feature_fn(X_tmp).astype(int)

        cv = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        score = cross_val_score(
            self.model,
            X_tmp,
            y,
            scoring=self.metric,
            cv=cv,
            n_jobs=-1
        ).mean()

        return score

    # --------------------------------------------------
    def safe_discovery(self, X, y, candidates):
        """
        ekle → dene → çıkar
        """
        if self.base_score is None:
            self.baseline_score(X, y)

        for name, fn in candidates.items():
            score = self.evaluate_single_feature(X, y, name, fn)

            if score > self.base_score + self.min_gain:
                self.kept_features.append(name)
                self.feature_defs[name] = fn
                print(f"[KEEP] {name}  +{score - self.base_score:.5f}")

        return self.kept_features

    # --------------------------------------------------
    def greedy_interactions(self, X, y, max_depth=3):
        """
        forward greedy interactions:
        kept_i & candidate_j
        """
        new_features = {}

        for kept_name in self.kept_features:
            kept_fn = self.feature_defs[kept_name]

            for other_name, other_fn in self.feature_defs.items():
                if kept_name == other_name:
                    continue

                new_name = f"{kept_name}__AND__{other_name}"

                def combined(df, f1=kept_fn, f2=other_fn):
                    return f1(df) & f2(df)

                new_features[new_name] = combined

        for name, fn in new_features.items():
            score = self.evaluate_single_feature(X, y, name, fn)

            if score > self.base_score + self.min_gain:
                self.kept_features.append(name)
                self.feature_defs[name] = fn
                self.base_score = score
                print(f"[GREEDY KEEP] {name}")

        return self.kept_features

    # --------------------------------------------------
    def prune_features(self, X, y):
        """
        çıkar → dene
        """
        pruned = []

        for fname in list(self.kept_features):
            X_tmp = X.copy()

            for other in self.kept_features:
                if other != fname:
                    X_tmp[other] = self.feature_defs[other](X_tmp).astype(int)

            cv = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
            score = cross_val_score(
                self.model,
                X_tmp,
                y,
                scoring=self.metric,
                cv=cv,
                n_jobs=-1
            ).mean()

            if score < self.base_score:
                pruned.append(fname)
                self.kept_features.remove(fname)
                print(f"[PRUNE] {fname}")

        return pruned

    # --------------------------------------------------
    def transform(self, X):
        X_new = X.copy()
        for name in self.kept_features:
            X_new[name] = self.feature_defs[name](X_new).astype(int)
        return X_new


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

gen = IndicatorFeatureGenerator(
    model=LGBMClassifier(),
    metric="roc_auc"
)

candidates = gen.generate_atomic_candidates(
    X,
    numeric_cols,
    categorical_cols,
    binary_cols
)

gen.safe_discovery(X, y, candidates)
gen.greedy_interactions(X, y)
gen.prune_features(X, y)

X_new = gen.transform(X)
