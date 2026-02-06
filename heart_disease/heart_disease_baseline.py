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

oof_X = pd.read_csv("./data/oof_train.csv" )
# print(X.head())
# print(y.head())


X["f1"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Sex"] == 1)
X["f2"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f3"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f4"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)

X["f5"] =   (X["Sex"] == 1) & (X["Chest pain type"] == 3) & (X["EKG results"] == 2)

X["f6"] = (X["Thallium"] == 3) & (X["Age"] < 53.00)

# X["ind_13"] =   (X["Sex"] == 0 ) & ( X["Chest pain type"] > 0 ) 

# X["ind_14"] =   (X["Thallium"] > 3 ) & ( X["FBS over 120"] ) & (X["EKG results"] > 0 )


# # X["ind_15"] = (X["Age"] < 37 ) & (X["Thallium"]==7) & ( X["Chest pain type"]==4 )
# # X["ind_16"] = (X["Age"] < 37 ) & (X["Thallium"]==3) & ( X["EKG results"]==2 )
# # X["ind_17"] = (X["Age"] < 37 ) & ( X["Chest pain type"]==2 ) & ( X["EKG results"]==2 )
# # X["ind_18"] = (X["Age"] > 69 ) & (X["Thallium"]==7) & ( X["Chest pain type"]==3 )
# # X["ind_19"] = (X["Age"] > 69 ) & (X["Thallium"]==3) & ( X["Chest pain type"]==3 )

# X["ind_20"] = (X["Age"] < 37 ) & (X["Thallium"]==7) & ( X["Sex"] )
# X["ind_21"] = (X["Age"] < 37 ) & ( X["Chest pain type"]==4 ) & ( X["Exercise angina"] )
# X["ind_22"] = (X["Age"] < 37 ) & ( X["Chest pain type"]==4 ) & ( X["FBS over 120"] )
# X["ind_23"] = (X["Age"] < 37 ) & ( X["Chest pain type"]==2 ) & ( X["Sex"] )
# X["ind_24"] = (X["Age"] < 37 ) & ( X["Chest pain type"]==2 ) & ( X["Exercise angina"] )





#print( X[ ["ind_"+str(i) for i in range(1,15)] ].corr() )

###

test["f1"] = (test["Age"] < 37.00) & (test["Thallium"] == 7) & (test["Sex"] == 1)
test["f2"] = (test["Age"] < 37.00) & (test["Thallium"] == 7) & (test["Exercise angina"] == 1)
test["f3"] = (test["Age"] > 69.00) & (test["Thallium"] == 7) & (test["Exercise angina"] == 1)
test["f4"] = (test["Age"] > 69.00) & (test["Thallium"] == 7) & (test["FBS over 120"] == 1)

test["f5"] =   (test["Sex"] == 1) & (test["Chest pain type"] == 3) & (test["EKG results"] == 2)

test["f6"] = (test["Thallium"] == 3) & (test["Age"] < 53.00)

# test["ind_1"] = (test["Age"] > 40 ) & ( test["Exercise angina"] )
# test["ind_2"] = (test["Age"] > 50) &  (test["FBS over 120"] )
# test["ind_3"] = ( test["Exercise angina"] ) &  (test["ST depression"] > 1 )
# test["ind_4"] = ( test["Exercise angina"] ) & (test["FBS over 120"] )

# test["ind_5"] = (test["ind_4"] ) & ( test["EKG results"] > 0 )

# test["ind_6"] = test["ind_3"] &  test["ind_4"] & (test["Thallium"] == 3)

# test["ind_7"] = ( test["EKG results"] > 0 ) & ( test["Chest pain type"]==3 )
# test["ind_8"] = ( test["EKG results"] > 0 ) & ( test["Chest pain type"]==2 )

# test["ind_9"] =  ( test["Number of vessels fluro"] > 0 ) & (test["Age"] < 40) 

# test["ind_10"] =  (test["Sex"] ==1 )  & (test["Age"] > 40 ) & ( test["Cholesterol"] > 300 )
# test["ind_11"] =  (test["Sex"] ==1 )  & (test["Age"] > 40 ) & ( test["BP"] > 180 )

# test["ind_12"] =   (test["ST depression"] > 2.5 ) & ( test["Slope of ST"] >= 2 ) & (test["Age"] > 50   )

# test["ind_13"] =   (test["Sex"] == 0 ) & ( test["Chest pain type"] > 0 ) 

# test["ind_14"] =   (test["Thallium"] > 3 ) & ( test["FBS over 120"] ) & (test["EKG results"] > 0 )


# # test["ind_15"] = (test["Age"] < 37 ) & (test["Thallium"]==7) & ( test["Chest pain type"]==4 )
# # test["ind_16"] = (test["Age"] < 37 ) & (test["Thallium"]==3) & ( test["EKG results"]==2 )
# # test["ind_17"] = (test["Age"] < 37 ) & ( test["Chest pain type"]==2 ) & ( test["EKG results"]==2 )

# # test["ind_18"] = (test["Age"] > 69 ) & (test["Thallium"]==7) & ( test["Chest pain type"]==3 )
# # test["ind_19"] = (test["Age"] > 69 ) & (test["Thallium"]==3) & ( test["Chest pain type"]==3 )

# test["ind_20"] = (test["Age"] < 37 ) & (test["Thallium"]==7) & ( test["Sex"] )
# test["ind_21"] = (test["Age"] < 37 ) & ( test["Chest pain type"]==4 ) & ( test["Exercise angina"] )
# test["ind_22"] = (test["Age"] < 37 ) & ( test["Chest pain type"]==4 ) & ( test["FBS over 120"] )
# test["ind_23"] = (test["Age"] < 37 ) & ( test["Chest pain type"]==2 ) & ( test["Sex"] )
# test["ind_24"] = (test["Age"] < 37 ) & ( test["Chest pain type"]==2 ) & ( test["Exercise angina"] )



# for i in range(1,25):
#     X["ind_"+str(i)] = X["ind_"+str(i)].astype(int) 


# for i in range(1,25):
#     test["ind_"+str(i)] = test["ind_"+str(i)].astype(int) 

##

#X["oof"] = oof_X["xgb"]

#     # "xgb": xgb,
#     # "lgbm": lgbm,
#     # "catc": catc,
#     # "rfc": rfc,
#     # "hgbc": hgbc,
#     # "gbc": gbc,
#     # "ada": ada


categorical_cols = ["Thallium","Chest pain type","EKG results"]

preprocess = make_column_transformer(
    (OneHotEncoder(), categorical_cols),
    remainder="passthrough"
)


X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075,random_state=42,stratify=y)


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

    # x_train_proc = preprocess.fit_transform(X_train)
    # X_valid_proc = preprocess.transform(X_valid)

    model = LGBMClassifier(**params)
    model.fit(X_train,y_train)

    #y_preds = model.predict(X_valid_proc)
    y_proba = model.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, y_proba)
    
    
    return score

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# print("Best ROC AUC:", study.best_value)
# print("Best params:", study.best_params)

best_params = {'n_estimators': 5613, 'max_depth': 5, 'num_leaves': 254, 'learning_rate': 0.013335190455676582, 'min_child_samples': 76, 'subsample': 0.6149990828235753, 'colsample_bytree': 0.1804665283299537, 'reg_alpha': 0.6399219443487499, 'reg_lambda': 0.8460546810433024}
best_params_2 = {'n_estimators': 724, 'max_depth': 2, 'num_leaves': 153, 'min_child_samples': 99, 'learning_rate': 0.1387114580881059, 'subsample': 0.37549286841241186, 'colsample_bytree': 0.9077375200328026, 'reg_alpha': 0.6578963730687483, 'reg_lambda': 0.28960307157515247}
best_params_3 = {'n_estimators': 7025, 'max_depth': 22, 'num_leaves': 15, 'learning_rate': 0.012752427604733264, 'min_child_samples': 34, 'subsample': 0.6090293649529753, 'colsample_bytree': 0.12166593548807814, 'reg_alpha': 0.0038796943982102212, 'reg_lambda': 0.3101264295152337}
best_params_4 = {'n_estimators': 1632, 'learning_rate': 0.17175168294090398, 'max_depth': 2, 'min_child_weight': 4.763727616534478, 'gamma': 0.3489319192846541, 'subsample': 0.8825152879554319, 'colsample_bytree': 0.8948997823632672, 'reg_alpha': 3.0520520893575527, 'reg_lambda': 2.3573646452813284}
best_params_5 = {'learning_rate': 0.21993613614736332, 'num_leaves': 300, 'max_depth': 5, 'min_child_samples': 109, 'subsample': 0.5998364348999323, 'colsample_bytree': 0.9323188541883586, 'reg_alpha': 0.06461329411023, 'reg_lambda': 5.592155905490778, 'n_estimators': 6157}

best_model = LGBMClassifier(**best_params , verbose=-1)

# x_train_proc = preprocess.fit_transform(X_train)
# X_valid_proc = preprocess.transform(X_valid)

best_model.fit(X_train,y_train)

y_preds = best_model.predict(X_valid)
y_proba = best_model.predict_proba(X_valid)[:, 1]

print({
"accuracy": accuracy_score(y_valid, y_preds),
"precision":precision_score(y_valid, y_preds),
"recall":recall_score(y_valid, y_preds),
"f1":f1_score(y_valid, y_preds),
"roc_auc":roc_auc_score(y_valid, y_proba),
"pr_auc":average_precision_score(y_valid,y_proba)
})

probas = best_model.predict_proba( test.drop("id",axis=1) )[:,1]

result = pd.DataFrame({"id":test["id"] , "Heart Disease": probas  })
result.to_csv("result.csv",index=False)