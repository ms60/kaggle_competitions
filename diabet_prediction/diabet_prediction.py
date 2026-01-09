import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from lightgbm import  LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV ,  train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder , StandardScaler , MinMaxScaler , FunctionTransformer

from sklearn.metrics import mean_absolute_error, r2_score , accuracy_score , classification_report ,  precision_score, recall_score, f1_score, roc_auc_score, average_precision_score



#pd.set_option('display.max_columns', None)

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print(train.head())

print(train.isnull().sum())
print(test.isnull().sum())

print(train.describe().T)

cat_ordinal_cols = ["education_level","income_level","smoking_status"]

print(train["education_level"].value_counts())
print(train["income_level"].value_counts())
print(train["smoking_status"].value_counts())

num_cols = ["age","alcohol_consumption_per_week","physical_activity_minutes_per_week","diet_score","sleep_hours_per_day","screen_time_hours_per_day","bmi","waist_to_hip_ratio","systolic_bp","heart_rate",
            "cholesterol_total","hdl_cholesterol","ldl_cholesterol","triglycerides"]

cat_ordinal_cols = ["education_level","income_level","smoking_status"]
cat_nominal_cols = ["employment_status","gender","ethnicity"]

preprocess = make_column_transformer(
    (OrdinalEncoder(categories=[["No formal","Highschool","Graduate","Postgraduate"],
                                ["Low","Lower-Middle","Middle","Upper-Middle","High"],
                                ["Never","Former","Current"]]),cat_ordinal_cols),
    (OneHotEncoder(handle_unknown='ignore'),cat_nominal_cols),
    (StandardScaler(),num_cols),
    remainder="drop"
)

model = LGBMClassifier(
    objective="binary",
    random_state=60,
    n_jobs=1,        # tüm CPU yerine 1 kullan → paralellik bazen RandomizedSearchCV ile sorun çıkarıyor
    verbose=-1       # console logları kapat
)

pipe = make_pipeline(
    preprocess,
    model

)
param_dist = {
    'lgbmclassifier__num_leaves': [15, 31, 63],
    'lgbmclassifier__max_depth': [-1, 5, 10, 15],
    'lgbmclassifier__learning_rate': [0.01, 0.05, 0.1],
    'lgbmclassifier__n_estimators': [100, 300, 500],
    'lgbmclassifier__min_child_samples': [5, 10],
    'lgbmclassifier__subsample': [0.6, 0.8, 1.0],               # row sampling
    'lgbmclassifier__colsample_bytree': [ 0.8, 1.0],        # feature sampling
    'lgbmclassifier__reg_alpha': [0, 0.1, 0.5],             # L1 regularization
    'lgbmclassifier__reg_lambda': [0, 0.1, 0.5],            # L2 regularization
    'lgbmclassifier__boosting_type': ['gbdt', 'dart'],           # 'goss' da eklenebilir
    'lgbmclassifier__class_weight': [None, 'balanced'],
}


rs = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=7,         # 10 kombinasyon → 1-2 dk sürer
    scoring="roc_auc",
    cv=3,              # 2-fold → hız kazanır
    verbose=2,
    n_jobs=1,          # paralellik kapalı → loop riskini azaltır
    random_state=42
)



X_train , X_test  , y_train , y_test = train_test_split(train.drop(["id","diagnosed_diabetes"] , axis =1) , train["diagnosed_diabetes"] , test_size=0.2, random_state=42)

rs.fit(X_train,y_train)
best_pipe = rs.best_estimator_
#preds = best_pipe.predict(X_test)


preds = best_pipe.predict(X_test)
probas = best_pipe.predict_proba(X_test)[:, 1]

print({
    "accuracy": accuracy_score(y_test, preds),
    "precision":precision_score(y_test, preds),
    "recall":recall_score(y_test, preds),
    "f1":f1_score(y_test, preds),
    "roc_auc":roc_auc_score(y_test, probas),
    "pr_auc":average_precision_score(y_test,probas)
})

pred_test = best_pipe.predict( test.drop("id",axis=1) )

result  = pd.DataFrame({"id":test["id"].to_list() , "diagnosed_diabetes":pred_test})
result.to_csv("result.csv",index=False)