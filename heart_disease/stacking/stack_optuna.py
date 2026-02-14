import optuna
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

meta_X = pd.read_csv("meta_X.csv")
meta_test = pd.read_csv("meta_test.csv")

print(meta_X.corr())

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})



logreg_ridge = LogisticRegression(
    penalty="l2",        # Ridge
    C=0.01,               # Regularization gücü (küçük C = daha güçlü)
    solver="lbfgs",      # default ve stabil
    max_iter=2000,
    n_jobs=-1
)

X_train , X_valid , y_train , y_valid = train_test_split(meta_X,y,test_size=0.075, shuffle=True ,stratify=y)

def objective(trial):

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",      # GPU varsa "gpu_hist"
        "n_estimators": trial.suggest_int("n_estimators", 300, 5000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "verbosity": 0,
        #"early_stopping_rounds":100,

    }

    model = XGBClassifier(**params)
    model.fit(X_train,y_train)

    #y_preds = model.predict(X_valid_proc)
    y_proba = model.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, y_proba)
    
    
    return score

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=75)

# cv = StratifiedKFold(n_splits=5 , shuffle=True , random_state=42)
# scores = cross_val_score(
#     logreg_ridge,
#     meta_X, y,
#     cv=cv,
#     scoring="roc_auc",
#     n_jobs=-1
# )

# print(scores)
# print( scores.mean() )





from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from tqdm import tqdm



class ManualWeightedEnsemble:
    def __init__(self, n_trials=5000, random_state=42, verbose=True):
        self.n_trials = n_trials
        self.random_state = random_state
        self.verbose = verbose
        self.best_weights = None
        self.best_score = -np.inf

    def _normalize(self, weights):
        weights = np.maximum(weights, 0)
        return weights / weights.sum()

    def fit(self, meta_X: pd.DataFrame, y: np.ndarray):
        """
        meta_X : OOF dataframe (n_samples, n_models)
        y      : true labels
        """
        rng = np.random.default_rng(self.random_state)
        X = meta_X.values

        iterator = range(self.n_trials)
        if self.verbose:
            iterator = tqdm(iterator, desc="Optimizing Weights")

        for _ in iterator:
            weights = rng.random(X.shape[1])
            weights = self._normalize(weights)

            preds = np.dot(X, weights)
            score = roc_auc_score(y, preds)

            if score > self.best_score:
                self.best_score = score
                self.best_weights = weights

                if self.verbose:
                    iterator.set_postfix(
                        best_auc=round(self.best_score, 6)
                    )

        print("\nBest AUC:", self.best_score)
        print("Best Weights:", self.best_weights)

    def predict(self, meta_X: pd.DataFrame):
        return np.dot(meta_X.values, self.best_weights)

    def get_weights(self):
        return self.best_weights


from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = ManualWeightedEnsemble(
    n_trials=3000,
    random_state=42,
    verbose=True
)

from sklearn.base import is_classifier
print(is_classifier(model))

# scores = cross_val_score(
#     model,
#     meta_X,
#     y,
#     cv=cv,
#     scoring="roc_auc",
#     n_jobs=-1
# )

# print("CV AUC Scores:", scores)
# print("Mean AUC:", scores.mean())


model.fit(meta_X, y)

train_pred = model.predict(meta_X)
test_pred = model.predict(meta_test)

print("Final Train AUC:",
        roc_auc_score(y, train_pred))


result = pd.DataFrame( { "id":test["id"] , "Heart Disease":test_pred } )
result.to_csv("result.csv",index=False)