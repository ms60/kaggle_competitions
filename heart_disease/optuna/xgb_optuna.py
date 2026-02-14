import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

# X["f1"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Sex"] == 1)
# X["f2"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
# X["f3"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
# X["f4"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)

# X["f5"] =   (X["Sex"] == 1) & (X["Chest pain type"] == 3) & (X["EKG results"] == 2)

# X["f6"] = (X["Thallium"] == 3) & (X["Age"] < 53.00)

print(X.head())

X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075, shuffle=True ,stratify=y)

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

# print("Best ROC AUC:", study.best_value)
# print("Best params:", study.best_params)
        
        
         
    
#best_params = {"tree_method": "hist", "eval_metric": "auc","objective": "binary:logistic",'n_estimators': 1926, 'learning_rate': 0.09013092119907051, 'max_depth': 2, 'min_child_weight': 7.875479116424961, 'gamma': 1.2524846771386975, 'subsample': 0.6636247422982167, 'colsample_bytree': 0.6334914652793887, 'reg_alpha': 0.40242893994715645, 'reg_lambda': 1.1887606704896005}
best_params = {"tree_method": "hist", "eval_metric": "auc","objective": "binary:logistic",'n_estimators': 4982, 'learning_rate': 0.09774724244098591, 'max_depth': 2, 'min_child_weight': 2.425776370513967, 'gamma': 3.045608819266778, 'subsample': 0.5830931356703783, 'colsample_bytree': 0.8245260385891402, 'reg_alpha': 2.9801695331430467, 'reg_lambda': 4.483112097021695}

best_model = XGBClassifier(**best_params)


skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(best_model,
                X,
                y,
                cv = skf,
                scoring="roc_auc",
                n_jobs=-1
                )

print(scores)
print(scores.mean())