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

print(X.head())
print(y.head())


X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)

nominal_cols = ["Sex","Thallium"] 

preprocess = make_column_transformer(
    (TargetEncoder(cols=nominal_cols, smoothing=5),nominal_cols),
    remainder="passthrough"
)

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

def objective(trial):

    params= {
        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
        'max_depth': trial.suggest_int('max_depth', 3, 32),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'class_weight': None,  # dilersen 'balanced' da ekleyebilirsin
        'random_state': 42,
        'verbosity': -1
    }

    X_train_proc = preprocess.fit_transform(X_train,y_train)
    X_valid_proc = preprocess.transform(X_valid)

    model = LGBMClassifier(**params)
    model.fit(X_train_proc,y_train)

    #y_preds = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid_proc)[:, 1]

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