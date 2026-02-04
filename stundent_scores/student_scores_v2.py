import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer 
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor , early_stopping , log_evaluation 
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder , StandardScaler 

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV ,  train_test_split
from sklearn.metrics import mean_absolute_error, r2_score , root_mean_squared_error

import optuna

from category_encoders import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

train = pd.read_csv("./data/train.csv")

print(train.head())

X = train.drop(["id","exam_score"] , axis=1).copy()
y = train["exam_score"]

#ordinal mapping
X["internet_access"] = X["internet_access"].map({"yes":1 , "no":0})
X["sleep_quality"] = X["sleep_quality"].map({"poor":-2 , "average":0.01 , "good":2 })
X["facility_rating"] = X["facility_rating"].map({"low":-5 , "medium":0.1 , "high":5 })
X["exam_difficulty"] = X["exam_difficulty"].map({"easy":0 , "moderate":1 , "hard":2 })


X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075, random_state=42)



cat_nominal = ["gender","course","study_method"]
cat_ordinal = ["internet_access","sleep_quality","facility_rating","exam_difficulty"]
num_cols = ["age","study_hours","class_attendance","sleep_hours"]






preprocess = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , cat_nominal ),
    (StandardScaler(),num_cols),
    (StandardScaler(),cat_ordinal),
    remainder="drop"
)

X_train_proc = preprocess.fit_transform(X_train)
X_valid_proc = preprocess.transform(X_valid)




# mi = mutual_info_regression(X_train_proc, y_train)
# mi_scores = pd.Series(mi, index=preprocess.get_feature_names_out()).sort_values(ascending=False)
# print(mi_scores)



# model = LGBMRegressor(
#     n_jobs=-1,
#     verbosity=0,
#     random_state=42
# )

# model.fit(X_train_proc,y_train)


# r = permutation_importance(
#     model,
#     X_valid_proc,
#     y_valid,
#     n_repeats=10,
#     random_state=42,
#     scoring="neg_root_mean_squared_error"
# )

# importances = pd.Series(r.importances_mean, index=preprocess.get_feature_names_out()).sort_values(ascending=False)
# print("-"*100)
# print(importances)


####################################################################







features_to_drop = ["gender","course","exam_difficulty","internet_access"]
X_filtered = X.copy()
print(X_filtered.head())

X_filtered["study_efficiency"] = X_filtered["study_hours"] * (X_filtered["class_attendance"] / 100.0)
X_filtered["sleep_efficiency"] = X_filtered["sleep_hours"] * X_filtered["sleep_quality"]
X_filtered["student_discipline_score"] = 0.4 * X_filtered["study_hours"] + 0.3 * X_filtered["class_attendance"] +0.3 * X_filtered["sleep_efficiency"]
X_filtered["facility_study_interaction"] = X_filtered["study_hours"] * X_filtered["facility_rating"]
X_filtered["low_attendance_flag"] = X_filtered["class_attendance"] < 75.0
X_filtered["sleep_deprivation_flag"] = (X_filtered["sleep_hours"] < 6 ) & ( X_filtered["study_hours"] > 5 )

X_filtered["study_hours_squared"] = X_filtered["study_hours"] * X_filtered["study_hours"]
X_filtered["study_hours_sqrt"] =  np.sqrt(X_filtered["study_hours"]) 
X_filtered["study_hours_log"] =  np.log1p(X_filtered["study_hours"]) 

X_filtered["study_hours_times_attendance"] = X_filtered["study_hours"] * X_filtered["class_attendance"]
X_filtered["study_hours_times_sleep"] = X_filtered["study_hours"] * X_filtered["sleep_hours"]

X_filtered['high_study_flag'] = (X_filtered['study_hours'] >= 7).astype(int)

print(X_filtered.isnull().sum())

ridge_pipe = make_pipeline(
    
        ( TargetEncoder(cols=cat_nominal, smoothing=5)),
        ( Ridge(alpha=1.0) )
    
)

kf = KFold(n_splits=3, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))

for train_idx, valid_idx in kf.split(X_filtered):
    X_train, X_valid = X_filtered.iloc[train_idx], X_filtered.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    ridge_pipe.fit(X_train, y_train)
    oof_preds[valid_idx] = ridge_pipe.predict(X_valid)

X_filtered["ridge_oof"] = oof_preds





cat_nominal = ["study_method","gender","course","internet_access"] # 
cat_ordinal = ["sleep_quality","facility_rating","exam_difficulty"] # 
num_cols = ["ridge_oof","age","study_hours","class_attendance","sleep_hours","study_efficiency","sleep_efficiency","student_discipline_score","facility_study_interaction","low_attendance_flag","sleep_deprivation_flag","study_hours_squared","study_hours_sqrt","study_hours_log","study_hours_times_attendance","study_hours_times_sleep","high_study_flag"] # 




preprocess_filtered = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , cat_nominal ),
    (StandardScaler(),num_cols),
    (StandardScaler(),cat_ordinal),
    remainder="drop"
)


X_train , X_valid , y_train , y_valid = train_test_split(X_filtered,y,test_size=0.070, random_state=42)

print(X_train.columns)
print(X_filtered.columns)

X_train_proc = preprocess_filtered.fit_transform(X_train)
X_valid_proc = preprocess_filtered.transform(X_valid)


def objective(trial):

    # params = {
    #     #'device': 'gpu',  # GPU acceleration
    #     "objective": "regression",
    #     "metric": "rmse",
    #     "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]), # ,"dart"
    #     "verbosity": -1,
    #     "force_row_wise": True,
    #     "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.9, log=True),
    #     "num_leaves": trial.suggest_int("num_leaves", 10, 512),
    #     "max_depth": trial.suggest_int("max_depth", 3, 32),
    #     "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
    #     "subsample": trial.suggest_float("subsample", 0.2, 1.0),
    #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    #     "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    #     "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    #     "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
    #     "random_state": 42,
    #     "n_jobs": -1
    # }
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",  # veya "gpu_hist" (GPU varsa)
        "eval_metric": "rmse",
        "booster": trial.suggest_categorical("booster", ["gbtree"]), #, "dart"
        'early_stopping_rounds': 100,
        
        # Öğrenme oranı ve derinlik
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.9, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 32),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        
        # Düzenlileştirme
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),   # L2
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),     # L1
        
        # Alt örnekleme
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        
        # DART için özel dropout parametreleri (booster=dart olduğunda aktif)
        "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.5),
        "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.5),
        
        # Ağaç sayısı
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
        "n_jobs": -1,
        "random_state": 42,
        
    }

    #model = LGBMRegressor(**params)
    model = XGBRegressor(**params)

    model.fit(
        X_train_proc, y_train,
        eval_set=[(X_valid_proc, y_valid)],
        verbose=False
    )

    y_pred = model.predict(X_valid_proc)
    rmse = root_mean_squared_error(y_valid, y_pred)

    return rmse




study = optuna.create_study(
    direction="minimize",
    study_name="lgb_exam_score"
)

study.optimize(objective, n_trials=100)

best_params = study.best_params

best_model = XGBRegressor(**best_params)

best_model.fit(
    X_train_proc, y_train,
    eval_set=[(X_valid_proc, y_valid)],
    verbose=False
)


y_pred = best_model.predict(X_valid_proc)
rmse = root_mean_squared_error(y_valid, y_pred)
print("Best RMSE on validation:", rmse)



# test_params = {'boosting_type': 'gbdt', 'learning_rate': 0.04652296188819206, 'num_leaves': 216, 'max_depth': 5, 'min_child_samples': 230, 'subsample': 0.9933406757691426, 'colsample_bytree': 0.26465493457945716, 'reg_alpha': 0.003176408880973185, 'reg_lambda': 0.10726491808899498, 'n_estimators': 7085}

# test_model = LGBMRegressor(**test_params)

# test = pd.read_csv("./data/test.csv")

# test["internet_access"] = test["internet_access"].map({"yes":1 , "no":0})
# test["sleep_quality"] = test["sleep_quality"].map({"poor":0 , "average":1 , "good":2 })
# test["facility_rating"] = test["facility_rating"].map({"low":0 , "medium":1 , "high":2 })
# test["exam_difficulty"] = test["exam_difficulty"].map({"easy":0 , "moderate":1 , "hard":2 })

# test["study_efficiency"] = test["study_hours"] * (test["class_attendance"] / 100.0)
# test["sleep_efficiency"] = test["sleep_hours"] * test["sleep_quality"]
# test["student_discipline_score"] = 0.4 * test["study_hours"] + 0.3 * test["class_attendance"] +0.3 * test["sleep_efficiency"]
# test["facility_study_interaction"] = test["study_hours"] * test["facility_rating"]
# test["low_attendance_flag"] = test["class_attendance"] < 75.0
# test["sleep_deprivation_flag"] = (test["sleep_hours"] < 6 ) & ( test["study_hours"] > 5 )

# test["study_hours_squared"] = test["study_hours"] * test["study_hours"]
# test["study_hours_sqrt"] =  np.sqrt(test["study_hours"]) 
# test["study_hours_log"] =  np.log1p(test["study_hours"]) 

# test["study_hours_times_attendance"] = test["study_hours"] * test["class_attendance"]
# test["study_hours_times_sleep"] = test["study_hours"] * test["sleep_hours"]

# test['high_study_flag'] = (test['study_hours'] >= 7).astype(int)

# print(X_filtered.columns)

# ridge_pipe.fit(X_filtered.drop("ridge_oof",axis=1), y)
# test["ridge_oof"] = ridge_pipe.predict(test.drop("id",axis=1))

# print(test.head())

# test_proc = preprocess_filtered.transform(test.drop("id",axis=1))


# print(X_filtered.head())


# test_model.fit(
#     X_train_proc, y_train,
#     eval_set=[(X_valid_proc, y_valid)],
#     eval_metric="rmse",
#     callbacks=[early_stopping(200), log_evaluation(0)]
# )


# test_preds = test_model.predict(test_proc, num_iteration=test_model.best_iteration_)

# result = pd.DataFrame( { "id":test["id"] , "exam_score":test_preds } )
# result.to_csv("result.csv",index=False)