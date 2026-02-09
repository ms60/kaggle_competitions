from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score , accuracy_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier, XGBRegressor

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

X["f1"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Sex"] == 1)
X["f2"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f3"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f4"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)

X["f5"] =   (X["Sex"] == 1) & (X["Chest pain type"] == 3) & (X["EKG results"] == 2)

X["f6"] = (X["Thallium"] == 3) & (X["Age"] < 53.00)

# X["f7"] = (X["Age"] > 68.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)
# X["f8"] = (X["Age"] > 65.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)



test["f1"] = (test["Age"] < 37.00) & (test["Thallium"] == 7) & (test["Sex"] == 1)
test["f2"] = (test["Age"] < 37.00) & (test["Thallium"] == 7) & (test["Exercise angina"] == 1)
test["f3"] = (test["Age"] > 69.00) & (test["Thallium"] == 7) & (test["Exercise angina"] == 1)
test["f4"] = (test["Age"] > 69.00) & (test["Thallium"] == 7) & (test["FBS over 120"] == 1)

test["f5"] =   (test["Sex"] == 1) & (test["Chest pain type"] == 3) & (test["EKG results"] == 2)

test["f6"] = (test["Thallium"] == 3) & (test["Age"] < 53.00)

# test["f7"] = (test["Age"] > 68.00) & (test["Thallium"] == 7) & (test["FBS over 120"] == 1)
# test["f8"] = (test["Age"] > 65.00) & (test["Thallium"] == 7) & (test["FBS over 120"] == 1)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    
lgbm_params = {'verbosity':-1,'objective': 'binary' , 'metric':'auc','boosting_type': 'gbdt', 'n_estimators': 6405, 'learning_rate': 0.030752124591604243, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.620293411878579, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}
lgbm_model = LGBMClassifier(**lgbm_params)
lgbm_oof_X = np.zeros( len(X) )
lgbm_oof_test = np.zeros( len(test) )

log_model = LogisticRegression(penalty="l2",C=1.0,solver="lbfgs",max_iter=1000,n_jobs=-1)
log_oof_X =  np.zeros( len(X) )
log_oof_test = np.zeros( len(test) )

xgb_model = XGBClassifier()
xgb_oof_X = np.zeros( len(X) )
xgb_oof_test = np.zeros( len(test) )

for train_index , validation_index in skf.split(X,y):
    X_train  = X.iloc[train_index]
    y_train = y.iloc[train_index]

    X_validation = X.iloc[validation_index]
    y_validation = y.iloc[validation_index]

    lgbm_model.fit( X_train,y_train )
    lgbm_oof_X[validation_index] =  lgbm_model.predict_proba(X_validation)[:, 1]

    # log_model.fit(X_train,y_train)
    # log_oof_X[validation_index] = log_model.predict_proba(X_validation)[:, 1]

    xgb_model.fit(X_train , y_train)
    xgb_oof_X[validation_index] = xgb_model.predict_proba(X_validation)[:, 1]

lgbm_model_test = clone(lgbm_model)
#log_model_test = clone(log_model)
xgb_model_test = clone(xgb_model)

for train_index , validation_index in skf.split(X,y):
    X_train  = X.iloc[train_index]
    y_train = y.iloc[train_index]

    X_validation = X.iloc[validation_index]
    y_validation = y.iloc[validation_index]

    lgbm_model_test.fit(X_train,y_train)
    lgbm_oof_test += lgbm_model_test.predict_proba(test.drop("id", axis=1))[:,1] / skf.n_splits

    # log_model_test.fit(X_train,y_train)
    # log_oof_test += log_model_test.predict_proba(test.drop("id", axis=1))[:,1] / skf.n_splits

    xgb_model_test.fit(X_train,y_train)
    xgb_oof_test += xgb_model_test.predict_proba(test.drop("id", axis=1))[:,1] / skf.n_splits

meta_model = RidgeClassifier()
meta_X = pd.DataFrame( { "lgbm":lgbm_oof_X , "xgb":xgb_oof_X } ) # , "logreg":log_oof_X 
meta_test = pd.DataFrame( {"lgbm":lgbm_oof_test , "xgb":xgb_oof_test  } ) # "logreg":log_oof_test

cv = StratifiedKFold(n_splits=5 , shuffle=True , random_state=42)
scores = cross_val_score(
    meta_model,
    meta_X, y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print(scores)
print( scores.mean() )


