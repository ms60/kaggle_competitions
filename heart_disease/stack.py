from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score , accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

oof_X = pd.read_csv("./data/oof_train.csv" )
oof_test = pd.read_csv("./data/oof_test.csv" )

selected = ["ada","xgb","nb_dist"]


X_train , X_valid , y_train , y_valid = train_test_split(oof_X[selected],y,test_size=0.075, random_state=42)


print(oof_X.corr())

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

    y_preds = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, y_proba)

    print({
    "accuracy": accuracy_score(y_valid, y_preds),
    "precision":precision_score(y_valid, y_preds),
    "recall":recall_score(y_valid, y_preds),
    "f1":f1_score(y_valid, y_preds),
    "roc_auc":roc_auc_score(y_valid, y_proba),
    "pr_auc":average_precision_score(y_valid,y_proba)
    })
        
    
    return score

# def objective(trial):
#     alpha = trial.suggest_float("alpha", 1e-4, 100, log=True)
    
#     meta = Ridge(alpha=alpha, random_state=42)
#     meta.fit(X_train,y_train)
#     y_preds = meta.predict(X_valid)
#     #y_proba = meta.predict_proba(X_valid)[:, 1]
#     score = roc_auc_score(y_valid, y_preds)#root_mean_squared_error(y_valid, y_preds)
#     #print( roc_auc_score(y_valid, y_preds) )

#     return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best ROC AUC:", study.best_value)
print("Best params:", study.best_params)

# best_model = Ridge(alpha=0.4047299221154111 ,random_state=42 )
# best_model.fit(X_train,y_train)

# preds = best_model.predict( oof_test[selected] )

# result = pd.DataFrame( {"id":test["id"] , "Heart Disease": preds} )
# result.to_csv("stack.csv",index=False)