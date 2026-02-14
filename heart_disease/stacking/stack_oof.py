from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score , accuracy_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from xgboost import XGBClassifier, XGBRegressor

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

numeric_cols = [
    "ST depression",
    "Age",  "Cholesterol",
    "Max HR", "BP","Slope of ST"
]

categorical_cols = [
    "Thallium", "Chest pain type"
]

ordinal_cols = ["EKG results","Number of vessels fluro"]

binary_cols = [
    "Exercise angina","Sex", "FBS over 120" 
]

categorical_cols_catboost = [
    "EKG results","Thallium", "Chest pain type","Slope of ST","Number of vessels fluro","Exercise angina","Sex", "FBS over 120" 
]

preprocess_v1 = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore") , categorical_cols+ ordinal_cols),
    #(TargetEncoder(cols=nominal_cols, smoothing=5) , nominal_cols ),
    (StandardScaler() , numeric_cols  ),
    remainder="passthrough"

)

preprocess_v2 = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore") , categorical_cols),
    #(TargetEncoder(cols=nominal_cols, smoothing=5) , nominal_cols ),
    (StandardScaler() , numeric_cols + ordinal_cols ),
    remainder="passthrough"

)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    
lgbm_params = {'verbosity':-1,'objective': 'binary' , 'metric':'auc','boosting_type': 'gbdt', 'n_estimators': 6405, 'learning_rate': 0.030752124591604243, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.620293411878579, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}
lgbm_model = LGBMClassifier(**lgbm_params)
lgbm_oof_X = np.zeros( len(X) )
lgbm_oof_test = np.zeros( len(test) )


xgb_params =  {"tree_method": "hist", "eval_metric": "auc","objective": "binary:logistic",'n_estimators': 4982, 'learning_rate': 0.09774724244098591, 'max_depth': 2, 'min_child_weight': 2.425776370513967, 'gamma': 3.045608819266778, 'subsample': 0.5830931356703783, 'colsample_bytree': 0.8245260385891402, 'reg_alpha': 2.9801695331430467, 'reg_lambda': 4.483112097021695}
xgb_model = XGBClassifier(**xgb_params)
xgb_oof_X = np.zeros( len(X) )
xgb_oof_test = np.zeros( len(test) )

catboost_params = {"thread_count": -1,"verbose": 0,"early_stopping_rounds": 200,"iterations": 2000,"eval_metric": "AUC","loss_function": "Logloss",'learning_rate': 0.07121073409644245, 'depth': 4, 'l2_leaf_reg': 1.347328885548022, 'random_strength': 2.5208976316842433, 'subsample': 0.9627839306440812, 'one_hot_max_size': 12, 'max_ctr_complexity': 1}
catboost_model = CatBoostClassifier(**catboost_params , cat_features=categorical_cols_catboost)
catboost_oof_X = np.zeros( len(X) )
catboost_oof_test = np.zeros(len(test))


logreg_ridge_v1 = LogisticRegression(
    penalty="l2",        # Ridge
    C=0.009125641392882999,               # Regularization gücü (küçük C = daha güçlü)
    solver="lbfgs",      # default ve stabil
    max_iter=1000,
    n_jobs=-1
)

logreg_ridge_v1_oof_X = np.zeros( len(X) )
logreg_ridge_v1_oof_test = np.zeros(len(test))

#

logreg_ridge_v2 = LogisticRegression(
    penalty="l2",        # Ridge
    C=0.24425776679024996,               # Regularization gücü (küçük C = daha güçlü)
    solver="lbfgs",      # default ve stabil
    max_iter=1000,
    n_jobs=-1
)

logreg_ridge_v2_oof_X = np.zeros( len(X) )
logreg_ridge_v2_oof_test = np.zeros(len(test))

#---
logreg_lasso_liblinear_v1 = LogisticRegression(
    penalty="l1",        # Lasso
    C=24.273648109254104,
    solver="liblinear", # veya "saga"
    max_iter=5000
)

logreg_lasso_liblinear_v1_oof_X = np.zeros( len(X) )
logreg_lasso_liblinear_v1_oof_test = np.zeros(len(test))

#

logreg_lasso_liblinear_v2 = LogisticRegression(
    penalty="l1",        # Lasso
    C=0.380498218569746,
    solver="liblinear", # veya "saga"
    max_iter=5000
)

logreg_lasso_liblinear_v2_oof_X = np.zeros( len(X) )
logreg_lasso_liblinear_v2_oof_test = np.zeros(len(test))

#---

logreg_lasso_saga_v2 = LogisticRegression(
    penalty="l1",        # Lasso
    C=0.004458034011942508,
    solver="saga", # veya "saga"
    max_iter=500
)

logreg_lasso_saga_v2_oof_X = np.zeros( len(X) )
logreg_lasso_saga_v2_oof_test = np.zeros(len(test))

#

logreg_lasso_saga_v1 = LogisticRegression(
    penalty="l1",        # Lasso
    C=8.552003705570856,
    solver="saga", # veya "saga"
    max_iter=100
)

logreg_lasso_saga_v1_oof_X = np.zeros( len(X) )
logreg_lasso_saga_v1_oof_test = np.zeros(len(test))

#-----

logreg_elastic_v1 = LogisticRegression(
    penalty="elasticnet",
    C=45.13119106050762,
    l1_ratio=0.5,       # 0 → Ridge, 1 → Lasso
    solver="saga",      # şart
    max_iter=100,
    n_jobs=-1
)

logreg_elastic_v1_oof_X = np.zeros( len(X) )
logreg_elastic_v1_oof_test = np.zeros(len(test))

logreg_elastic_v2 = LogisticRegression(
    penalty="elasticnet",
    C=6.515177362156471,
    l1_ratio=0.5,       # 0 → Ridge, 1 → Lasso
    solver="saga",      # şart
    max_iter=100,
    n_jobs=-1
)

logreg_elastic_v2_oof_X = np.zeros( len(X) )
logreg_elastic_v2_oof_test = np.zeros(len(test))



for train_index , validation_index in skf.split(X,y):
    X_train  = X.iloc[train_index]
    y_train = y.iloc[train_index]
    

    X_validation = X.iloc[validation_index]
    y_validation = y.iloc[validation_index]

    X_train_proc_v1 = preprocess_v1.fit_transform(X_train)
    X_valid_proc_v1 = preprocess_v1.transform(X_validation)

    X_train_proc_v2 = preprocess_v2.fit_transform(X_train)
    X_valid_proc_v2 = preprocess_v2.transform(X_validation)

    #---

    lgbm_model.fit( X_train,y_train )
    lgbm_oof_X[validation_index] =  lgbm_model.predict_proba(X_validation)[:, 1]

    xgb_model.fit(X_train , y_train)
    xgb_oof_X[validation_index] = xgb_model.predict_proba(X_validation)[:, 1]

    catboost_model.fit(X_train,y_train , cat_features= categorical_cols_catboost)
    catboost_oof_X[validation_index] = catboost_model.predict_proba(X_validation)[:,1]

    logreg_ridge_v1.fit(X_train_proc_v1,y_train)
    logreg_ridge_v1_oof_X[validation_index] = logreg_ridge_v1.predict_proba(X_valid_proc_v1)[:,1]

    logreg_ridge_v2.fit(X_train_proc_v2,y_train)
    logreg_ridge_v2_oof_X[validation_index] = logreg_ridge_v2.predict_proba(X_valid_proc_v2)[:,1]

    logreg_lasso_liblinear_v1.fit(X_train_proc_v1,y_train)
    logreg_lasso_liblinear_v1_oof_X[validation_index] = logreg_lasso_liblinear_v1.predict_proba(X_valid_proc_v1)[:,1]

    logreg_lasso_liblinear_v2.fit(X_train_proc_v2,y_train)
    logreg_lasso_liblinear_v2_oof_X[validation_index] = logreg_lasso_liblinear_v2.predict_proba(X_valid_proc_v2)[:,1]

    logreg_lasso_saga_v1.fit(X_train_proc_v1,y_train)
    logreg_lasso_saga_v1_oof_X[validation_index] = logreg_lasso_saga_v1.predict_proba(X_valid_proc_v1)[:,1]

    logreg_lasso_saga_v2.fit(X_train_proc_v2,y_train)
    logreg_lasso_saga_v2_oof_X[validation_index] = logreg_lasso_saga_v2.predict_proba(X_valid_proc_v2)[:,1]

    logreg_elastic_v1.fit(X_train_proc_v1,y_train)
    logreg_elastic_v1_oof_X[validation_index] = logreg_elastic_v1.predict_proba(X_valid_proc_v1)[:,1]

    logreg_elastic_v2.fit(X_train_proc_v2,y_train)
    logreg_elastic_v2_oof_X[validation_index] = logreg_elastic_v2.predict_proba(X_valid_proc_v2)[:,1]


lgbm_model_test = clone(lgbm_model)
xgb_model_test = clone(xgb_model)
catboost_model_test = clone(catboost_model)
logreg_ridge_v1_test = clone(logreg_ridge_v1)
logreg_ridge_v2_test = clone(logreg_ridge_v2)

logreg_lasso_liblinear_v1_test = clone(logreg_lasso_liblinear_v1)
logreg_lasso_liblinear_v2_test = clone(logreg_lasso_liblinear_v2)

logreg_lasso_saga_v1_test = clone(logreg_lasso_saga_v1)
logreg_lasso_saga_v2_test = clone(logreg_lasso_saga_v2)

logreg_elastic_v1_test = clone(logreg_elastic_v1)
logreg_elastic_v2_test = clone(logreg_elastic_v2)


for train_index , validation_index in skf.split(X,y):
    X_train  = X.iloc[train_index]
    y_train = y.iloc[train_index]

    X_validation = X.iloc[validation_index]
    y_validation = y.iloc[validation_index]

    X_train_proc_v1 = preprocess_v1.fit_transform(X_train)
    test_proc_v1 = preprocess_v1.transform(test.drop("id",axis=1))


    X_train_proc_v2 = preprocess_v2.fit_transform(X_train)
    test_proc_v2 = preprocess_v2.transform(test.drop("id",axis=1))
    

    lgbm_model_test.fit(X_train,y_train)
    lgbm_oof_test += lgbm_model_test.predict_proba(test.drop("id", axis=1))[:,1] / skf.n_splits

    xgb_model_test.fit(X_train,y_train)
    xgb_oof_test += xgb_model_test.predict_proba(test.drop("id", axis=1))[:,1] / skf.n_splits

    catboost_model_test.fit(X_train,y_train, cat_features= categorical_cols_catboost)
    catboost_oof_test += catboost_model_test.predict_proba(test.drop("id", axis=1))[:,1] / skf.n_splits

    logreg_ridge_v1_test.fit(X_train_proc_v1,y_train)
    logreg_ridge_v1_oof_test += logreg_ridge_v1_test.predict_proba(test_proc_v1)[:,1] / skf.n_splits

    logreg_ridge_v2_test.fit(X_train_proc_v2,y_train)
    logreg_ridge_v2_oof_test += logreg_ridge_v2_test.predict_proba(test_proc_v2)[:,1] / skf.n_splits

    logreg_lasso_liblinear_v1_test.fit(X_train_proc_v1,y_train)
    logreg_lasso_liblinear_v1_oof_test += logreg_lasso_liblinear_v1_test.predict_proba(test_proc_v1)[:,1] / skf.n_splits

    logreg_lasso_liblinear_v2_test.fit(X_train_proc_v2,y_train)
    logreg_lasso_liblinear_v2_oof_test += logreg_lasso_liblinear_v2_test.predict_proba(test_proc_v2)[:,1] / skf.n_splits

    logreg_lasso_saga_v1_test.fit(X_train_proc_v1,y_train)
    logreg_lasso_saga_v1_oof_test += logreg_lasso_saga_v1_test.predict_proba(test_proc_v1)[:,1] / skf.n_splits

    logreg_lasso_saga_v2_test.fit(X_train_proc_v2,y_train)
    logreg_lasso_saga_v2_oof_test += logreg_lasso_saga_v2_test.predict_proba(test_proc_v2)[:,1] / skf.n_splits

    logreg_elastic_v1_test.fit(X_train_proc_v1,y_train)
    logreg_elastic_v1_oof_test += logreg_elastic_v1_test.predict_proba(test_proc_v1)[:,1] / skf.n_splits

    logreg_elastic_v2_test.fit(X_train_proc_v2,y_train)
    logreg_elastic_v2_oof_test += logreg_elastic_v2_test.predict_proba(test_proc_v2)[:,1] / skf.n_splits


meta_model = LGBMClassifier()
meta_X = pd.DataFrame( { "lgbm":lgbm_oof_X , "xgb":xgb_oof_X , "catboost":catboost_oof_X ,
                        "logreg_ridge_v1":logreg_ridge_v1_oof_X , "logreg_ridge_v2":logreg_ridge_v2_oof_X,
                          "logreg_lasso_liblinear_v1":logreg_lasso_liblinear_v1_oof_X , "logreg_lasso_liblinear_v2":logreg_lasso_liblinear_v2_oof_X,
                             "logreg_lasso_saga_v1":logreg_lasso_saga_v1_oof_X , "logreg_lasso_saga_v2":logreg_lasso_saga_v2_oof_X,
                                 "logreg_elastic_v1":logreg_elastic_v1_oof_X , "logreg_elastic_v2":logreg_elastic_v2_oof_X  } ) 


meta_test = pd.DataFrame( { "lgbm":lgbm_oof_test , "xgb":xgb_oof_test , "catboost":catboost_oof_test ,
                        "logreg_ridge_v1":logreg_ridge_v1_oof_test , "logreg_ridge_v2":logreg_ridge_v2_oof_test,
                          "logreg_lasso_liblinear_v1":logreg_lasso_liblinear_v1_oof_test , "logreg_lasso_liblinear_v2":logreg_lasso_liblinear_v2_oof_test,
                             "logreg_lasso_saga_v1":logreg_lasso_saga_v1_oof_test , "logreg_lasso_saga_v2":logreg_lasso_saga_v2_oof_test,
                                 "logreg_elastic_v1":logreg_elastic_v1_oof_test , "logreg_elastic_v2":logreg_elastic_v2_oof_test  } )  


meta_X.to_csv("meta_X.csv",index=False)
meta_test.to_csv("meta_test.csv",index=False)


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

# meta_model.fit(meta_X , y)
# predicts = meta_model.predict(meta_test)

# result = pd.DataFrame( { "id":test["id"] , "Heart Disease":predicts } )
# result.to_csv("result.csv",index=False)

