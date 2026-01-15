import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer , KNNImputer , IterativeImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.compose import make_column_transformer , ColumnTransformer

from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder , StandardScaler , MinMaxScaler , FunctionTransformer
from sklearn.pipeline import make_pipeline

from lightgbm import LGBMRegressor , LGBMClassifier

from sklearn.metrics import mean_absolute_error, r2_score , accuracy_score , classification_report ,  precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import optuna

import re

import lightgbm as lgb
from sklearn.metrics import log_loss



def seperate_cabin(text):
    match = re.match(r"([a-z]+)([0-9]+)", text, re.I)
    if match:
        items = match.groups()
    return items

def handle_age(X):
    X = pd.Series(X[:,0].squeeze())
    bins = [0, 4, 8, 14, 18, 45, 60, np.inf]
    groups = [
        'AGE_0_4', 'AGE_4_8', 'AGE_8_14',
        'AGE_14_18', 'AGE_18_45', 'AGE_45_60', 'AGE_60_PLUS'
    ]
    return pd.DataFrame(  pd.cut(X, bins=bins, labels=groups , include_lowest=True) , columns=["AGE_GROUP"] )

"""
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

"""

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")


print(train.head())

print(train.isnull().sum())
print(train.shape)

######

print(train["Ticket"].sort_values())
print(train["Ticket"].nunique())

print(train.describe().T)


##################
#Feature Engineering

##Ticket
print( train["Ticket"].str.split(" ").apply(lambda row: len(row)).max() ) # max 3 
print(train.apply(lambda col : col["Ticket"].split(" ") , axis = 1 ) )
train["Ticket_F"] = train.apply(lambda col : col["Ticket"].split(" ")[0]  , axis = 1 )
train["Ticket_S"] = train.apply(lambda col : col["Ticket"].split(" ")[1] if len(col["Ticket"].split(" "))>1 else None , axis = 1 )
train["Ticket_T"] = train.apply(lambda col : col["Ticket"].split(" ")[2] if len(col["Ticket"].split(" "))>2 else None , axis = 1 )

train["Ticket_S"] = train.apply(lambda col: col["Ticket_F"] if col["Ticket_S"] is  None else col["Ticket_S"],axis = 1 )
train["Ticket_F"] = train.apply(lambda col: None if col["Ticket_F"]==col["Ticket_S"] else col["Ticket_F"] ,axis = 1 )



##Ticket

test["Ticket_F"] = test.apply(lambda col : col["Ticket"].split(" ")[0]  , axis = 1 )
test["Ticket_S"] = test.apply(lambda col : col["Ticket"].split(" ")[1] if len(col["Ticket"].split(" "))>1 else None , axis = 1 )
test["Ticket_T"] = test.apply(lambda col : col["Ticket"].split(" ")[2] if len(col["Ticket"].split(" "))>2 else None , axis = 1 )

test["Ticket_S"] = test.apply(lambda col: col["Ticket_F"] if col["Ticket_S"] is  None else col["Ticket_S"],axis = 1 )
test["Ticket_F"] = test.apply(lambda col: None if col["Ticket_F"]==col["Ticket_S"] else col["Ticket_F"] ,axis = 1 )


##Age


train = train[ train["Embarked"].notnull() ]


#print(train["Cabin"].apply(lambda row: seperate_cabin(row ) ))

# print(train["Cabin"].value_counts())

# print(train.head())

# print( train["Age"].squeeze() )
#####################

X_train , X_test  , y_train , y_test = train_test_split(train.drop(["PassengerId","Survived"] , axis =1) , train["Survived"] , test_size=0.2, random_state=60)

cat_ordinal_cols = ["Pclass","SibSp","Parch"]
cat_nominal_cols = ["Sex","Embarked"]
num_cols = ["Fare"]


age_pipeline = make_pipeline(
    SimpleImputer( strategy="median", add_indicator=True),
    FunctionTransformer(handle_age,validate=False),
    OneHotEncoder(handle_unknown='ignore')
)




ct_train = make_column_transformer(
    (age_pipeline, ["Age"]),  
    (StandardScaler() , num_cols),
    (OneHotEncoder(handle_unknown='ignore'), cat_nominal_cols),
    remainder='drop'
)

X_train_proc = ct_train.fit_transform(X_train)
X_test_proc = ct_train.transform(X_test)

def objective(trial):
    param = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type":"dart",
        #"boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 5000),
        "random_state": 42,
        "verbosity": -1,
    }
    
    model = lgb.LGBMClassifier(**param)
    model.fit(
        X_train_proc, y_train,
        eval_set=[(X_test_proc, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=100)],

    )
    
    preds = model.predict_proba(X_test_proc)[:, 1]  # pozitif sınıf olasılıkları
    auc = roc_auc_score(y_test, preds)
    return auc  # maximize edeceğiz

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, timeout=3600)

print("Best params:", study.best_trial.params)
print("Best logloss:", study.best_trial.value)

lgbm_model = lgb.LGBMClassifier(**study.best_trial.params)
lgbm_model.fit(X_train_proc,y_train)



y_pred = lgbm_model.predict(X_test_proc)  # sınıf tahmini
y_proba = lgbm_model.predict_proba(X_test_proc)[:, 1]

print({
    "accuracy": accuracy_score(y_test, y_pred),
    "precision":precision_score(y_test, y_pred),
    "recall":recall_score(y_test, y_pred),
    "f1":f1_score(y_test, y_pred),
    "roc_auc":roc_auc_score(y_test, y_proba),
    "pr_auc":average_precision_score(y_test,y_proba)
})


X_result_transformed = ct_train.transform(test)
y_result = lgbm_model.predict(X_result_transformed)  

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": y_result
})

submission.to_csv("result_optuna.csv", index=False)

"""
{'accuracy': 0.8258426966292135, 'precision': 0.8461538461538461, 'recall': 0.7236842105263158, 'f1': 0.7801418439716312, 'roc_auc': 0.8782894736842104, 'pr_auc': 0.8619192257571586}
"""
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


def objective_xgb(trial):
    params = {
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "tree_method": "hist",  # GPU varsa "gpu_hist"
        "eval_metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 32),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 7000),
        # DART parametreleri, booster='dart' ise aktif
        "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.5),
        "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.5),
        #"use_label_encoder": False,
        'early_stopping_rounds': 150,
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train_proc, y_train,
        eval_set=[(X_test_proc, y_test)],
        verbose=False
    )

    preds_proba = model.predict_proba(X_test_proc)[:, 1]  # pozitif sınıf olasılığı
    auc = roc_auc_score(y_test, preds_proba)
    return auc  # Optuna direction='maximize' olacak

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=150, timeout=3600)

xgbm_model = XGBClassifier(**study_xgb.best_trial.params)
xgbm_model.fit(X_train_proc,y_train)


y_result_xgb = xgbm_model.predict(X_result_transformed)  

print("Best params study_xgb:", study_xgb.best_trial.params)
print("Best logloss study_xgb:", study_xgb.best_trial.value)

submission_xgb = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": y_result
})

submission_xgb.to_csv("result_optuna_xgb.csv", index=False)