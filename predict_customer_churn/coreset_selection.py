from itertools import combinations, product
import random
from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

import shap
from tqdm import tqdm
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)



train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

train["gender"] = train["gender"].map({"Female": 0, "Male": 1})
test["gender"] = test["gender"].map({"Female": 0, "Male": 1})

train["Partner"] = train["Partner"].map({"No": 0, "Yes": 1})
test["Partner"] = test["Partner"].map({"No": 0, "Yes": 1})

train["Dependents"] = train["Dependents"].map({"No": 0, "Yes": 1})
test["Dependents"] = test["Dependents"].map({"No": 0, "Yes": 1})

train["PhoneService"] = train["PhoneService"].map({"No": 0, "Yes": 1})
test["PhoneService"] = test["PhoneService"].map({"No": 0, "Yes": 1})

train["MultipleLines"] = train["MultipleLines"].map({"No": 0, "Yes": 1, "No phone service": 0})
test["MultipleLines"] = test["MultipleLines"].map({"No": 0, "Yes": 1, "No phone service": 0})

train["InternetService"] = train["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2})
test["InternetService"] = test["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2})

train["OnlineSecurity"] = train["OnlineSecurity"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["OnlineSecurity"] = test["OnlineSecurity"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["OnlineBackup"] = train["OnlineBackup"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["OnlineBackup"] = test["OnlineBackup"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["DeviceProtection"] = train["DeviceProtection"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["DeviceProtection"] = test["DeviceProtection"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["TechSupport"] = train["TechSupport"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["TechSupport"] = test["TechSupport"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["StreamingTV"] = train["StreamingTV"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["StreamingTV"] = test["StreamingTV"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["StreamingMovies"] = train["StreamingMovies"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["StreamingMovies"] = test["StreamingMovies"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["PaperlessBilling"] = train["PaperlessBilling"].map({"No": 0, "Yes": 1})
test["PaperlessBilling"] = test["PaperlessBilling"].map({"No": 0, "Yes": 1})

train["Churn"] = train["Churn"].map({"No": 0, "Yes": 1})

train = train.drop("id",axis=1)

for col in ["PaymentMethod","Contract"]:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")



#--------------------------------------------------

X= train.drop("Churn",axis=1)
y = train["Churn"]

X_test = test.drop("id",axis=1)

#------------------------------------------------

SUBSET_SIZE = 50_000

#-------

# method 1 : simple stratify

train_stratified, _ = train_test_split(
    train,
    train_size=SUBSET_SIZE,
    stratify=train["Churn"],
    random_state=42
)

X_stratified = train_stratified.drop("Churn", axis=1)
y_stratified = train_stratified["Churn"]

#--------

# method 2 : Clustering based sampling

from sklearn.cluster import MiniBatchKMeans

train_clustering_based = train.copy()

train_clustering_based_transformed = pd.get_dummies(train_clustering_based,columns=["PaymentMethod","Contract"] , dtype = int)

for col in ["MonthlyCharges","TotalCharges"]:
    train_clustering_based_transformed[col] = np.log1p(train_clustering_based_transformed[col])

for col in ["tenure","MonthlyCharges","TotalCharges"]:
    train_clustering_based_transformed[col] = StandardScaler().fit_transform(train_clustering_based_transformed[[col]])


print(train_clustering_based_transformed)

# mbk = MiniBatchKMeans(n_clusters=50000, batch_size=2048)
# mbk.fit(train_clustering_based_transformed)

mbk = MiniBatchKMeans(n_clusters=50000, batch_size=2048, random_state=42)
mbk.fit(train_clustering_based_transformed)


X_clusters = mbk.cluster_centers_

X_subset = pd.DataFrame(X_clusters, columns=train_clustering_based_transformed.columns)

print(X_subset)

#print(X_clustering_based )

def check_feature_importance_stability(X1,y1,X2,y2):
    model1 = XGBClassifier(**xgb_params)
    model1.fit(X1,y1,
               eval_set=[(X1, y1)],
               verbose=0
    )

    model2 = XGBClassifier(**xgb_params)
    model2.fit(X2,y2,
               eval_set=[(X2, y2)],
               verbose=0
    )

    feature_importances_1 = model1.feature_importances_
    feature_importances_2 = model2.feature_importances_

    corr = np.corrcoef(
        feature_importances_1,
        feature_importances_2
    )[0,1]
    return corr


from scipy.stats import ks_2samp

def check_ks(X1,X2):

    ks_scores = {}

    for col in X1.columns:
        ks = ks_2samp(X1[col], X2[col]).statistic
        ks_scores[col] = ks

    return np.mean(list(ks_scores.values())) 


#print(X_clustering_based.dtypes)

# print( check_ks( X , X_stratified ) )
# print( y.mean() ,y_stratified.mean() )
# print( check_feature_importance_stability(X,y,X_stratified,y_stratified) )




#---------------

#stats encoding
for col in ["PaymentMethod","Contract"]:
    stats = train_stratified.groupby(col)["Churn"].agg(["mean","std","count"])

    for s in stats.columns:
        X_stratified[f"{col}_{s}"] = X_stratified[col].map(stats[s]).astype("float32")
        

# rank encoding
for col in ["PaymentMethod","Contract"]:

    counts = X_stratified[col].value_counts()
    rank_map = counts.rank(method="dense", ascending=False)

    X_stratified[f"{col}_rank"] = X_stratified[col].map(rank_map).astype("int32")


def oof_target_encode(train, col, target, n_splits=5):

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 )
    oof = np.zeros(len(train))

    for tr_idx, val_idx in kf.split(train , train[target] ):

        tr = train.iloc[tr_idx]
        val = train.iloc[val_idx]

        means = tr.groupby(col)[target].mean()

        oof[val_idx] = val[col].map(means)

    return oof

# for col in ["gender","Partner","Dependents","PhoneService","InternetService","PaperlessBilling"]:
#     X[f"{col}_te"] = oof_target_encode(X, col, "Churn")
#     means = X.groupby(col)["Churn"].mean()
#     X_test[f"{col}_te"] = X_test[col].map(means)







#--------------------------------------------



xgb_params = {
    'n_estimators': 3000,      
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree':0.8,
    'max_bin':4000,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    #'early_stopping_rounds': 200,
    'device': 'cuda',
    
    'enable_categorical': True,
}

lgbm_params = {
    'n_estimators': 60000,
    'learning_rate': 0.002,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree':0.8,
    'max_bin':12000,
    'objective': 'binary',
    'metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    #'early_stopping_rounds': 200,
    #'device': 'cuda',
    'verbosity':-1,
}

te_cols = ["PaperlessBilling"] # ,"gender","Partner","Dependents","PhoneService","InternetService","PaperlessBilling"

# def manual_cv(model , X , y ):
#     X_train_1 , X_valid_1 , y_train_1 , y_valid_1 = train_test_split(X , y , test_size=0.075 , random_state=42 , stratify=y)
#     X_train_2, X_valid_2, y_train_2, y_valid_2 = train_test_split(X, y, test_size=0.1, random_state=60, stratify=y)
#     X_train_3, X_valid_3, y_train_3, y_valid_3 = train_test_split(X, y, test_size=0.2, random_state=31, stratify=y)

#     for col in te_cols:
#         X_train_1[f"{col}_te"] = oof_target_encode(X_train_1, col, "Churn")
#         means = X_train_1.groupby(col)["Churn"].mean()
#         X_valid_1[f"{col}_te"] = X_valid_1[col].map(means)

#         X_train_1 = X_train_1.drop(col,axis=1)
#         X_valid_1 = X_valid_1.drop(col,axis=1)
        


#     for col in te_cols:
#         X_train_2[f"{col}_te"] = oof_target_encode(X_train_2, col, "Churn")
#         means = X_train_2.groupby(col)["Churn"].mean()
#         X_valid_2[f"{col}_te"] = X_valid_2[col].map(means)

#         X_train_2 = X_train_2.drop(col,axis=1)
#         X_valid_2 = X_valid_2.drop(col,axis=1)


#     for col in te_cols:
#         X_train_3[f"{col}_te"] = oof_target_encode(X_train_3, col, "Churn")
#         means = X_train_3.groupby(col)["Churn"].mean()
#         X_valid_3[f"{col}_te"] = X_valid_3[col].map(means)

#         X_train_3 = X_train_3.drop(col,axis=1)
#         X_valid_3 = X_valid_3.drop(col,axis=1)



#     X_train_1 = X_train_1.drop("Churn",axis=1)
#     X_train_2 = X_train_2.drop("Churn",axis=1)
#     X_train_3 = X_train_3.drop("Churn",axis=1)


#     X_valid_1 = X_valid_1.drop("Churn",axis=1)
#     X_valid_2 = X_valid_2.drop("Churn",axis=1)
#     X_valid_3 = X_valid_3.drop("Churn",axis=1)


#     model1 = clone(model)
#     model2 = clone(model)
#     model3 = clone(model)

#     model1.fit(X_train_1, y_train_1,
#                #eval_set=[(X_valid_1, y_valid_1)],
#                #verbose=1000
#     )
#     model2.fit(X_train_2, y_train_2,
#                #eval_set=[(X_valid_2, y_valid_2)],
#                #verbose=1000
#     )
#     model3.fit(X_train_3, y_train_3,
#                #eval_set=[(X_valid_3, y_valid_3)],
#                #verbose=1000
#     )

#     y_pred1 = model1.predict_proba(X_valid_1)[:, 1]
#     y_pred2 = model2.predict_proba(X_valid_2)[:, 1]
#     y_pred3 = model3.predict_proba(X_valid_3)[:, 1]

#     score1 = roc_auc_score(y_valid_1, y_pred1)
#     score2 = roc_auc_score(y_valid_2, y_pred2)
#     score3 = roc_auc_score(y_valid_3, y_pred3)

#     print([score1, score2, score3])
#     print(np.mean([score1, score2, score3]))
#     print(np.std([score1, score2, score3]))

#     return [score1, score2, score3]




    




# model = XGBClassifier(**xgb_params)

# manual_cv(model , X_stratified , y_stratified)
# manual_cv(model , X_clustering_based , y_clustering_based)

# model = LGBMClassifier(**lgbm_params)

# manual_cv(model , train_stratified , y_stratified)




#print( check_ks( X , X_clustering_based ) )

#  X_stratified_3CV -> 0.9101627909690619  ,X_stratified_LB -> 0.91033 , X_LB -> 0.91495
#  X_stratified_FE_3CV -> 0.9103271521292317 ,