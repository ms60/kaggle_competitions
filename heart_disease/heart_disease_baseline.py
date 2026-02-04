from category_encoders import TargetEncoder
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



train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

# print(X.head())
# print(y.head())


X["ind_1"] = (X["Age"] > 40 ) & ( X["Exercise angina"] )
X["ind_2"] = (X["Age"] > 50) &  (X["FBS over 120"] )
X["ind_3"] = ( X["Exercise angina"] ) &  (X["ST depression"] > 1 )
X["ind_4"] = ( X["Exercise angina"] ) & (X["FBS over 120"] )

X["ind_5"] = (X["ind_4"] ) & ( X["EKG results"] > 0 )

X["ind_6"] = X["ind_3"] &  X["ind_4"] & (X["Thallium"] == 3)

X["ind_7"] = ( X["EKG results"] > 0 ) & ( X["Chest pain type"]==3 )
X["ind_8"] = ( X["EKG results"] > 0 ) & ( X["Chest pain type"]==2 )

X["ind_9"] =  ( X["Number of vessels fluro"] > 0 ) & (X["Age"] < 40) 

X["ind_10"] =  (X["Sex"] ==1 )  & (X["Age"] > 40 ) & ( X["Cholesterol"] > 300 )
X["ind_11"] =  (X["Sex"] ==1 )  & (X["Age"] > 40 ) & ( X["BP"] > 180 )


for i in range(1,12):
    X["ind_"+str(i)] = X["ind_"+str(i)].astype(int) 


X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)


# model = LGBMClassifier()
# model.fit(X_train,y_train)

# y_preds = model.predict(X_valid)
# y_proba = model.predict_proba(X_valid)[:, 1]

# print({
# "accuracy": accuracy_score(y_valid, y_preds),
# "precision":precision_score(y_valid, y_preds),
# "recall":recall_score(y_valid, y_preds),
# "f1":f1_score(y_valid, y_preds),
# "roc_auc":roc_auc_score(y_valid, y_proba),
# "pr_auc":average_precision_score(y_valid,y_proba)
# })


# test_proba = model.predict_proba(test.drop("id", axis=1))[:, 1]

# submission = pd.DataFrame({
#     "id": test["id"],
#     "Heart Disease": test_proba
# })
# submission.to_csv("submission.csv", index=False)
# submission.head()

        # "boosting_type": "gbdt",    
        # "force_row_wise": True,
        # # model params
        # "learning_rate": trial.suggest_float("learning_rate", 0.01 , 0.9, log=True),
        # "num_leaves": trial.suggest_int("num_leaves" ,10, 512),
        # "max_depth": trial.suggest_int("max_depth", 3, 16),
        # "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
        # "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        # "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        # "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        # "n_estimators":trial.suggest_int("n_estimators", 100 ,10000 ),
        # "random_state": 42,
        # "n_jobs": -1,
        # 'verbosity': -1

def objective(trial):

    params= {
    "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]),
    "objective": "binary",
    "metric": "auc",
    "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
    "num_leaves": trial.suggest_int("num_leaves", 10, 512),
    "max_depth": trial.suggest_int("max_depth", 3, 32),
    "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
    "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
    "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    "max_bin": trial.suggest_int("max_bin", 64, 512),
    "random_state": 42,
    "verbosity": -1,
    }



    model = LGBMClassifier(**params)
    model.fit(X_train,y_train)

    #y_preds = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, y_proba)
    
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best ROC AUC:", study.best_value)
print("Best params:", study.best_params)

#best_params = {'n_estimators': 4290, 'max_depth': 3, 'num_leaves': 39, 'learning_rate': 0.03530678249463069, 'min_child_samples': 48, 'subsample': 0.14215733250141216, 'colsample_bytree': 0.5462748555787986, 'reg_alpha': 0.33755662630401917, 'reg_lambda': 0.040551519519335555}

# best_model = LGBMClassifier(**best_params)
# best_model.fit(X_train,y_train)
# probas = best_model.predict_proba( test.drop("id",axis=1) )[:,1]

# result = pd.DataFrame({"id":test["id"] , "Heart Disease": probas  })
# result.to_csv("result.csv",index=False)