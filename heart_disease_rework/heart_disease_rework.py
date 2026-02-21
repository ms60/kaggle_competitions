from itertools import combinations,product
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import clone
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score, train_test_split, KFold , StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import mean_absolute_error, r2_score , accuracy_score , classification_report ,  precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder , StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier 
from category_encoders import TargetEncoder

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

print(X.head())
print(y.head())

print(X.shape)

X["f1"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Sex"] == 1)
X["f2"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f3"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f4"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)

X["f5"] =   (X["Sex"] == 1) & (X["Chest pain type"] == 3) & (X["EKG results"] == 2)

X["f6"] = (X["Thallium"] == 3) & (X["Age"] < 53.00)

for i in range(1,7):
    X["f"+str(i)] = X["f"+str(i)].astype(int) 

X_male = X[X["Sex"] == 1]
X_female = X[X["Sex"] == 0]

y_male = y[X["Sex"] == 1]
y_female = y[X["Sex"] == 0]

X_male = X_male.drop("Sex",axis = 1)
X_female = X_female.drop("Sex",axis = 1)



X_male_small, _, y_male_small, _ = train_test_split(
    X_male, y_male,
    train_size=80_000,
    stratify=y_male,
    shuffle=True,
    random_state=42
)


X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075, shuffle=True ,stratify=y)










def objective(trial):
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",  # veya "gpu_hist" (GPU varsa)
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree"]), #, "dart"
        #'early_stopping_rounds': 100,
        
        # Öğrenme oranı ve derinlik
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.9, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 32),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        
        # Düzenlileştirme
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),   # L2
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),     # L1
        
        # Alt örnekleme
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        
        # DART için özel dropout parametreleri (booster=dart olduğunda aktif)
        "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.5),
        "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.5),
        
        # Ağaç sayısı
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
        "n_jobs": -1,
        "random_state": 42,
        
    }


    model = XGBClassifier(**params)
    model.fit(X_train,y_train)

    #y_preds = model.predict(X_valid_proc)
    y_proba = model.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, y_proba)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=150)

print("Best ROC AUC:", study.best_value)
print("Best params:", study.best_params)

#{'boosting_type': 'gbdt', 'n_estimators': 5514, 'learning_rate': 0.029859877967825382, 'num_leaves': 235, 'max_depth': 3, 'min_child_samples': 13, 'min_child_weight': 0.8145170341294354, 'min_split_gain': 0.1079317737197928, 'subsample': 0.21948035297883112, 'colsample_bytree': 0.24449842837603247, 'reg_alpha': 0.004007727852920216, 'reg_lambda': 0.1405164647380011}
# 0.9492040376707642.