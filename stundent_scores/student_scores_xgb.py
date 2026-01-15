import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline

from lightgbm import LGBMRegressor , LGBMClassifier , early_stopping , log_evaluation
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV ,  train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder , StandardScaler , MinMaxScaler , FunctionTransformer

from sklearn.metrics import mean_absolute_error, r2_score , root_mean_squared_error
import optuna


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")



print(train.head())
print(train.shape)
print(train.dtypes)

print(train.isnull().sum()) # no missing column ez

## feature engineering

train["study_efficiency"] = train["study_hours"] * (train["class_attendance"] / 100.0)
test["study_efficiency"] = test["study_hours"] * (test["class_attendance"] / 100.0)

train["sleep_efficiency"] = train["sleep_hours"] * train["sleep_quality"].map({"poor":1,"average":2,"good":3})
test["sleep_efficiency"] = test["sleep_hours"] * test["sleep_quality"].map({"poor":1,"average":2,"good":3})

train["student_discipline_score"] = 0.4 * train["study_hours"] + 0.3 * train["class_attendance"] +0.3 * train["sleep_efficiency"]
test["student_discipline_score"] = 0.4 * test["study_hours"] + 0.3 * test["class_attendance"] +0.3 * test["sleep_efficiency"]


train["facility_study_interaction"] = train["study_hours"] * train["facility_rating"].map({"low":1,"medium":2,"high":3})
test["facility_study_interaction"] = test["study_hours"] * test["facility_rating"].map({"low":1,"medium":2,"high":3})

train["low_attendance_flag"] = train["class_attendance"] < 75.0
test["low_attendance_flag"] = test["class_attendance"] < 75.0

train["sleep_deprivation_flag"] = (train["sleep_hours"] < 6 ) & ( train["study_hours"] > 5 )
test["sleep_deprivation_flag"] = (test["sleep_hours"] < 6 ) & ( test["study_hours"] > 5 )

train["over_study_flag"] = train["study_hours"] > 8
test["over_study_flag"] = test["study_hours"] > 8

train["study_hours_squared"] = train["study_hours"] * train["study_hours"]
test["study_hours_squared"] = test["study_hours"] * test["study_hours"]

train["sleep_hours_squared"] = train["sleep_hours"] * train["sleep_hours"]
test["sleep_hours_squared"] = test["sleep_hours"] * test["sleep_hours"]

# study_hours_squared
# sleep_hours_squared

train["special_feature"] = (6*train.study_hours + 0.35*train.class_attendance + 1.5*train.sleep_hours +
                 5*(train.sleep_quality=='good') + -5*(train.sleep_quality=='poor') +
                 10*(train.study_method=='coaching') + 5*(train.study_method=='mixed') + 2*(train.study_method=='group study') + 1*(train.study_method=='online videos') +
                 4*(train.facility_rating=='high') + -4*(train.facility_rating=='low') )

test["special_feature"] = (6*test.study_hours + 0.35*test.class_attendance + 1.5*test.sleep_hours +
                 5*(test.sleep_quality=='good') + -5*(test.sleep_quality=='poor') +
                 10*(test.study_method=='coaching') + 5*(test.study_method=='mixed') + 2*(test.study_method=='group study') + 1*(test.study_method=='online videos') +
                 4*(test.facility_rating=='high') + -4*(test.facility_rating=='low') )


###

cat_ordinal_cols = ["sleep_quality","facility_rating","exam_difficulty","internet_access","study_method"]
cat_nominal_cols = ["gender","course","low_attendance_flag","sleep_deprivation_flag","over_study_flag","age"]
num_cols = ["study_hours","class_attendance","sleep_hours","study_efficiency","sleep_efficiency","student_discipline_score","facility_study_interaction","study_hours_squared","sleep_hours_squared","special_feature"]

target_col = ["exam_score"]

for col in cat_ordinal_cols:
    print(train[col].value_counts().index  )


print(train[num_cols+["exam_score"]].describe().T)

print("="*80)

for col in cat_nominal_cols:
    print( train[col].value_counts() / train.shape[0] )

#class imbalances : internet_access , course
print("="*80)
for col in cat_ordinal_cols:
    print( train[col].value_counts() / train.shape[0] )

#class imbalances: exam_difficulty


preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , cat_nominal_cols),
    (OrdinalEncoder(categories=[
        ["poor","average","good"],
        ["low","medium","high"],
        ["easy","moderate","hard"],
        ["no","yes"],
        ["self-study","online videos","group study","mixed","coaching"]
        ]),cat_ordinal_cols),
    (StandardScaler(),num_cols),
    remainder="drop"

)





X_train , X_test  , y_train , y_test = train_test_split(train.drop(["id","exam_score"] , axis =1) , train["exam_score"] , test_size=0.075, random_state=42)


X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

def objective(trial):

    params = {
    
        # model params
        "tree_method": "hist",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001 , 0.9, log=True),
        "num_leaves": trial.suggest_int("num_leaves" ,10, 512),
        "learning_rate": trial.suggest_float("learning_rate", 0.01 , 0.9, log=True),
        "num_leaves": trial.suggest_int("num_leaves" ,10, 512),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),   # L2
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),     # L1
        "n_estimators":trial.suggest_int("n_estimators", 500 ,10000 ),
        'random_state': 42,
        'eval_metric': 'rmse',
        "objective": "reg:squarederror",
        'early_stopping_rounds': 150,
        "n_jobs": -1,


    }




    model = XGBRegressor(**params)

    model.fit(
        X_train_proc,
        y_train,
        eval_set=[(X_test_proc, y_test)],
        verbose=1000
    )

    y_pred = model.predict(X_test_proc)
    rmse = root_mean_squared_error(y_test, y_pred)

    return rmse

study = optuna.create_study(
    direction="minimize",
    study_name="xgb_exam_score"
)

study.optimize(objective, n_trials=50)

xgb_params = {
    'n_estimators': 15000,
    'learning_rate': 0.007,
    'max_depth': 6,
    'subsample': 0.90,
    'num_parallel_tree': 2,
    'reg_lambda': 5,
    'colsample_bytree': 0.5, 
    'colsample_bynode': 0.7,
    'tree_method': 'hist',
    'random_state': 42,
    'early_stopping_rounds': 100,
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'device': 'cuda',
    'min_child_weight': 6
} 

best_params = study.best_params
#best_params = {'learning_rate': 0.03000270673879387, 'num_leaves': 106, 'max_depth': 6, 'min_child_samples': 72, 'subsample': 0.911567996350799, 'colsample_bytree': 0.7179754224465255, 'reg_alpha': 0.023972346751633882, 'reg_lambda': 0.6469720859882682, 'n_estimators': 4575}

final_model = XGBRegressor(
    **best_params,
    #objective="regression",

    n_jobs=-1
)

final_model.fit(
    X_train_proc,
    y_train,
    eval_set=[(X_test_proc, y_test)],
    verbose=1000

)

preds = final_model.predict( X_test_proc )


print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE",root_mean_squared_error(y_test,preds))
print("R2 :", r2_score(y_test, preds))

test_proc = preprocessor.transform(test.drop("id",axis=1))

preds_test = final_model.predict(test_proc)

result  = pd.DataFrame({"id":test["id"].to_list() , "exam_score":preds_test})
result.to_csv("result.csv",index=False)

#best_pipe = rs.best_estimator_
#preds = best_pipe.predict(X_test)
# preds = pipe.predict(X_test)




# print("MAE:", mean_absolute_error(y_test, preds))
# print("RMSE",root_mean_squared_error(y_test,preds))
# print("R2 :", r2_score(y_test, preds))
# ######

# #pred_test = best_pipe.predict( test.drop("id",axis=1) )
# pred_test = pipe.predict( test.drop("id",axis=1) )

# result  = pd.DataFrame({"id":test["id"].to_list() , "exam_score":pred_test})
# result.to_csv("result.csv",index=False)