import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from category_encoders import TargetEncoder

from itertools import combinations
import optuna

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from category_encoders import TargetEncoder

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import FunctionTransformer




#---------- Read data ----------
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis=1).copy()
y = X.pop("exam_score")

X["internet_access"] = X["internet_access"].map({"yes":1 , "no":0})
X["sleep_quality"] = X["sleep_quality"].map({"poor":-5 , "average":0.1 , "good":5 })
X["facility_rating"] = X["facility_rating"].map({"low":-4 , "medium":0.1 , "high":4 })
X["exam_difficulty"] = X["exam_difficulty"].map({"easy":1 , "moderate":2 , "hard":3 })

X["study_method"] = X["study_method"].map({
    'coaching': 10,
    'mixed': 5,
    'group study': 3,
    'online videos': 2,
    'self-study': 0.1
})


print(X.head())
print(y.head())

num_cols = ["age","study_hours","class_attendance","sleep_hours","study_method"]
ordinal_cols = ["sleep_quality","facility_rating","exam_difficulty"]
nominal_cols = ["gender","course","internet_access"]


print( X.groupby("study_method")["internet_access"].count() )


##FE
num_cols_all = num_cols[:]

#squares , log , sqrt
for col in num_cols:
    X[col + "_" + "squared" ] = X[col] * X[col]
    X[col + "_" + "sqrt" ] = np.sqrt( X[col] )
    X[col + "_" + "log" ] = np.log1p( X[col] )

    num_cols_all.append(col + "_" + "squared")
    num_cols_all.append(col + "_" + "sqrt")
    num_cols_all.append(col + "_" + "log")

# numeric interactions , multiply-division

for col1, col2 in combinations(num_cols, 2):
    X[col1 + "_multiply_" + col2] = X[col1] * X[col2]
    X[col1 + "_divide_" + col2] = X[col1] / X[col2]

    num_cols_all.append(col1 + "_multiply_" + col2)
    num_cols_all.append(col1 + "_divide_" + col2)



#ordinal interactions
for col1, col2 in combinations(ordinal_cols, 2):
    X[col1 + "_multiply_" + col2 ] = X[col1] * X[col2]
    X[col1 + "_divide_" + col2 ] = X[col1] / X[col2]

    num_cols_all.append(col1 + "_multiply_" + col2)
    num_cols_all.append(col1 + "_divide_" + col2)

#numeric - ordinal interactions

for col1 in num_cols:
    for col2 in ordinal_cols:
        X[col1 + "_multiply_" + col2 ] = X[col1] * X[col2]
        X[col1 + "_divide_" + col2 ] = X[col1] / X[col2]

        num_cols_all.append(col1 + "_multiply_" + col2)
        num_cols_all.append(col1 + "_divide_" + col2)



######################################################33
#squares , log , sqrt

print(X.head())

identity = FunctionTransformer(lambda x: x)

preprocess = make_column_transformer(
    (TargetEncoder(cols=nominal_cols,smoothing=10), nominal_cols),
    #( OneHotEncoder(handle_unknown='ignore') , nominal_cols ) ,
    ( StandardScaler() , num_cols_all ),
    remainder="passthrough"

)

pipe = make_pipeline(
    preprocess
    )

# # models = {
# #     "ridge": Ridge(alpha=1.0),
# #     "lasso": Lasso(alpha=0.1),
# #     "elastic": ElasticNet(alpha=0.1, l1_ratio=0.5),
# #     #"rf": RandomForestRegressor(random_state=42),
# #     "xgb":XGBRegressor(n_estimators=500),
# #     "lgbm":LGBMRegressor(n_estimators=500),
# # }

# # for name, model in models.items():
# #     pipe = make_pipeline(
# #         preprocess,
# #         model
# #     )

# #     scores = cross_val_score(
# #         pipe,
# #         X,
# #         y,
# #         cv=5,
# #         scoring="neg_root_mean_squared_error"
# #     )

# #     print(name, "RMSE:", -scores.mean())

# X_sfs = X[num_cols_all] 

# sfs = SequentialFeatureSelector(
#     estimator=pipe,
#     n_features_to_select=10,
#     direction="forward",
#     scoring="neg_root_mean_squared_error",
#     cv=5,
#     n_jobs=-1
# )

# sfs.fit(X_sfs, y)

# selected_features = X.columns[sfs.get_support()]
# print(selected_features)

X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.070, random_state=42)


# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# oof_ridge = np.zeros(len(X))
# test_preds_ridge = np.zeros((len(test), 10))
# orig_preds_ridge = np.zeros(len(X))

# ridge_alphas = np.logspace(-3, 3, 20)

# print("Training Ridge Regression")
# print("-" * 40)

# for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
#     X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
#     # Augment with original data
#     X_tr_aug = pd.concat([X_tr, X], axis=0)
#     y_tr_aug = pd.concat([y_tr, y], axis=0)
    
#     # Target encode categoricals
#     encoder = TargetEncoder(smooth='auto', target_type='continuous')
#     X_tr_enc = X_tr_aug.copy()
#     X_val_enc = X_val.copy()
#     X_test_enc = test.copy()
    
#     X_tr_enc[nominal_cols] = encoder.fit_transform(X_tr_aug[nominal_cols], y_tr_aug)
#     X_val_enc[nominal_cols] = encoder.transform(X_val[nominal_cols])
#     X_test_enc[nominal_cols] = encoder.transform(X_valid[nominal_cols])
    
#     # Fit Ridge
#     ridge = RidgeCV(alphas=ridge_alphas, cv=5, scoring='neg_root_mean_squared_error')
#     ridge.fit(X_tr_enc, y_tr_aug.values.ravel())
    
#     # Predictions (clipped to valid range)
#     oof_ridge[val_idx] = np.clip(ridge.predict(X_val_enc), 0, 100)
#     test_preds_ridge[:, fold - 1] = np.clip(ridge.predict(X_test_enc), 0, 100)
#     orig_preds_ridge += np.clip(ridge.predict(X_tr_enc.iloc[-len(X_original):]), 0, 100) / N_FOLDS
    
#     rmse = np.sqrt(mean_squared_error(y_val, oof_ridge[val_idx]))
#     print(f"Fold {fold:2d} | RMSE: {rmse:.6f}")

# ridge_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_ridge))
# print(f"\nRidge OOF RMSE: {ridge_oof_rmse:.6f}")

X_train_proc = pipe.fit_transform(X_train,y_train)
X_valid_proc = pipe.transform(X_valid)



def objective(trial):

    params = {
        #'device': 'gpu',  # GPU acceleration
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]), # ,"dart"
        "verbosity": -1,
        "force_row_wise": True,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.9, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 512),
        "max_depth": trial.suggest_int("max_depth", 3, 32),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
        "random_state": 42,
        "n_jobs": -1
    }


    model = LGBMRegressor(**params)
    #model = XGBRegressor(**params)

    model.fit(
        X_train_proc, y_train,
        eval_set=[(X_valid_proc, y_valid)],
        eval_metric="rmse",
        callbacks=[early_stopping(200), log_evaluation(0)]
    )

    y_pred = model.predict(X_valid_proc, num_iteration=model.best_iteration_)
    rmse = root_mean_squared_error(y_valid, y_pred)

    return rmse

study = optuna.create_study(
    direction="minimize",
    study_name="lgb_exam_score"
)

study.optimize(objective, n_trials=100)

best_params = study.best_params
print(best_params)

