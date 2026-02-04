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
from xgboost import XGBClassifier 
from category_encoders import TargetEncoder

from itertools import combinations


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
train_raw = train.copy()

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

print(X.head())
print(y.head())



# X["Age_flag"] = (X["Age"] > 40.0).astype(int)
# X["BP_flag"] = (X["BP"] > 150.0).astype(int)
# X["Max_HR_flag"] = (X["Max HR"] > 180).astype(int)
# X["Cholesterol_flag"] = (X["Cholesterol"] > 300.0).astype(int)

# X["EKG results"] = X["EKG results"].map({0:5 , 1 : -2 , 2:-5 })

nominal_cols = ["Sex","Chest pain type","Thallium"] 
num_cols = ["Age","BP","Cholesterol","Max HR","ST depression","Slope of ST",]
#te_cols = ["EKG results","Sex","Chest pain type","Thallium"]
ordinal_cols = ["EKG results","Number of vessels fluro"]
yes_or_no = ["FBS over 120","Exercise angina","Age_flag","BP_flag","Max_HR_flag","Cholesterol_flag"]

# for col in num_cols:
#     X[ col + "_squared" ] = X[col] * X[col]
#     X[ col + "_log"] = np.log1p( X[col] )
#     X[ col + "_sqrt" ] = np.sqrt( X[col] )

# for col in ordinal_cols:
#     X[ col + "_squared" ] = X[col] * X[col]
#     X[ col + "_log"] = np.log1p( X[col] )
#     X[ col + "_sqrt" ] = np.sqrt( X[col] )

# for col1, col2 in combinations(num_cols, 2):
#     X[col1 + "_multiply_" + col2] = X[col1] * X[col2]
#     X[col1 + "_divide_" + col2] = X[col1] / X[col2]

# for col1, col2 in combinations(ordinal_cols, 2):
#     X[col1 + "_multiply_" + col2] = X[col1] * X[col2]
#     X[col1 + "_divide_" + col2] = X[col1] / X[col2]

# for col1 in num_cols:
#     for col2 in ordinal_cols:
#         X[col1 + "_multiply_" + col2 ] = X[col1] * X[col2]
#         X[col1 + "_divide_" + col2 ] = X[col1] / X[col2]

print(X.head())

X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075, random_state=42,stratify=y)


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



    model = LGBMClassifier(**params)
    model.fit(X_train,y_train)

    #y_preds = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, y_proba)
    
    
    return score

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# print("Best ROC AUC:", study.best_value)
# print("Best params:", study.best_params)

best_params = {'n_estimators': 7025, 'max_depth': 22, 'num_leaves': 15, 'learning_rate': 0.012752427604733264, 'min_child_samples': 34, 'subsample': 0.6090293649529753, 'colsample_bytree': 0.12166593548807814, 'reg_alpha': 0.0038796943982102212, 'reg_lambda': 0.3101264295152337}
# best_model = LGBMClassifier(**best_params)
# best_model.fit(X_train,y_train)
# y_proba = best_model.predict_proba( X_valid )[:,1]
# print( roc_auc_score(y_valid, y_proba) )


# #best_params = study.best_params


# test["Age_flag"] = (test["Age"] > 40.0).astype(int)
# test["BP_flag"] = (test["BP"] > 150.0).astype(int)
# test["Max_HR_flag"] = (test["Max HR"] > 180).astype(int)
# test["Cholesterol_flag"] = (test["Cholesterol"] > 300.0).astype(int)

# test["EKG results"] = test["EKG results"].map({0:5 , 1 : -2 , 2:-5 })

# for col in num_cols:
#     test[ col + "_squared" ] = test[col] * test[col]
#     test[ col + "_log"] = np.log1p( test[col] )
#     test[ col + "_sqrt" ] = np.sqrt( test[col] )

# # for col in ordinal_cols:
# #     X[ col + "_squared" ] = X[col] * X[col]
# #     X[ col + "_log"] = np.log1p( X[col] )
# #     X[ col + "_sqrt" ] = np.sqrt( X[col] )

# for col1, col2 in combinations(num_cols, 2):
#     test[col1 + "_multiply_" + col2] = test[col1] * test[col2]
#     test[col1 + "_divide_" + col2] = test[col1] / test[col2]

# for col1, col2 in combinations(ordinal_cols, 2):
#     test[col1 + "_multiply_" + col2] = test[col1] * test[col2]
#     test[col1 + "_divide_" + col2] = test[col1] / test[col2]

# for col1 in num_cols:
#     for col2 in ordinal_cols:
#         test[col1 + "_multiply_" + col2 ] = test[col1] * test[col2]
#         test[col1 + "_divide_" + col2 ] = test[col1] / test[col2]




best_model = LGBMClassifier(**best_params)
best_model.fit(X_train,y_train)
probas = best_model.predict_proba( test.drop("id",axis=1) )[:,1]

result = pd.DataFrame({"id":test["id"] , "Heart Disease": probas  })
result.to_csv("result.csv",index=False)
