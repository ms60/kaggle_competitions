from catboost import CatBoostClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})


categorical_cols_catboost = [
    "EKG results","Thallium", "Chest pain type","Slope of ST","Number of vessels fluro","Exercise angina","Sex", "FBS over 120" 
]


X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075, shuffle=True ,stratify=y)

def objective(trial):

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 2000,
        "early_stopping_rounds": 200,
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,

        # --- Optimize edilenler ---
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),

        # categorical için önemli
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
        "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 3),

        # hız için genelde Plain öneriyorum
        "boosting_type": "Plain",
        "bootstrap_type": "Bernoulli",
    }

    model = CatBoostClassifier(**params , cat_features= categorical_cols_catboost   ) 

    #model = XGBClassifier(**params)
    model.fit(X_train,y_train)

    #y_preds = model.predict(X_valid_proc)
    y_proba = model.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, y_proba)
    
    
    return score


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=75)

# print("Best ROC AUC:", study.best_value)
# print("Best params:", study.best_params)
        

        # 
        # 
        # 
        # 
        # "random_seed": 42,
        # 
        # 


best_params = {"thread_count": -1,"verbose": 0,"early_stopping_rounds": 200,"iterations": 2000,"eval_metric": "AUC","loss_function": "Logloss",'learning_rate': 0.07121073409644245, 'depth': 4, 'l2_leaf_reg': 1.347328885548022, 'random_strength': 2.5208976316842433, 'subsample': 0.9627839306440812, 'one_hot_max_size': 12, 'max_ctr_complexity': 1}

best_params_2 = {"thread_count": -1,"verbose": 0,"early_stopping_rounds": 200,"iterations": 2000,"eval_metric": "AUC","loss_function": "Logloss",'learning_rate': 0.06646646222708427, 'depth': 4, 'l2_leaf_reg': 1.280908103829506, 'random_strength': 2.835473648571126, 'subsample': 0.855360986274129, 'one_hot_max_size': 15, 'max_ctr_complexity': 2}    
best_params_3 = {"thread_count": -1,"verbose": 0,"early_stopping_rounds": 200,"iterations": 2000,"eval_metric": "AUC","loss_function": "Logloss",'learning_rate': 0.04644434060974833, 'depth': 4, 'l2_leaf_reg': 13.315535600205463, 'random_strength': 0.7583574879305478, 'subsample': 0.8526177762348528, 'one_hot_max_size': 15, 'max_ctr_complexity': 2}


best_model = CatBoostClassifier(**best_params , cat_features=categorical_cols_catboost)


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