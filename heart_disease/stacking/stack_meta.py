from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

meta_X = pd.read_csv("meta_X_lgbm_1_lgbm_2_logreg_elasticnet_.csv")

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

print(meta_X.corr())

blend_oof = (
    0.40 * meta_X["lgbm_1"] +
    0.35 * meta_X["lgbm_2"] +
    0.25 * meta_X["logreg_elasticnet"]
)

cv_score = roc_auc_score(y, blend_oof)
print(cv_score)
