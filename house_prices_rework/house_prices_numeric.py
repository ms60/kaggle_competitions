from lightgbm import LGBMRegressor
import optuna
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from sklearn.compose import make_column_transformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler , TargetEncoder
from sklearn.metrics import  root_mean_squared_error
from xgboost import XGBRegressor


#plt.style.use("seaborn-v0_8-white")
#plt.subplots_adjust(wspace=0.4, hspace=0.5)

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

sortedColumns = train.drop("Id",axis=1).columns.tolist()
sortedColumns.sort()



X = train[sortedColumns]
y = X.pop("SalePrice")
y = np.log1p(y)

sortedColumns.remove("SalePrice")
X_test = test.drop("Id",axis=1)
X_test = X_test[sortedColumns]


print(X.head())
# print(y.head())
# print(X_test.head())

# pre actions

X["YearBuilt"] =  X["YearBuilt"] - X["YearBuilt"].min()
X["YearRemodAdd"] = X["YearRemodAdd"] - X["YearRemodAdd"].min()
X["GarageYrBlt"] = X["GarageYrBlt"] - X["GarageYrBlt"].min() 

X_test["YearBuilt"] =  X_test["YearBuilt"] - X_test["YearBuilt"].min()
X_test["YearRemodAdd"] = X_test["YearRemodAdd"] - X_test["YearRemodAdd"].min()
X_test["GarageYrBlt"] = X_test["GarageYrBlt"] - X_test["GarageYrBlt"].min() 



numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_features.remove("MoSold")
numeric_features.remove("YrSold")

categorical_features.append("MoSold")
categorical_features.append("YrSold")

X_numeric = X[numeric_features]
X_test = X_test[numeric_features]

zeroFlagList = [  "2ndFlrSF" , "3SsnPorch" , "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","EnclosedPorch","GarageArea","LowQualFinSF","MSSubClass","MasVnrArea","MiscVal","OpenPorchSF","PoolArea","ScreenPorch","WoodDeckSF","YearRemodAdd"]

log1pList = [ "1stFlrSF","2ndFlrSF","3SsnPorch","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF" , "EnclosedPorch","GarageArea","GarageYrBlt","GrLivArea","LotArea","LotFrontage","MasVnrArea","MiscVal","OpenPorchSF",
 "PoolArea","ScreenPorch","TotalBsmtSF","WoodDeckSF","YearBuilt","YearRemodAdd"]

targetEncodingList = ["BedroomAbvGr","Fireplaces","FullBath","HalfBath","KitchenAbvGr","LowQualFinSF","MSSubClass","OverallCond","OverallQual","TotRmsAbvGrd"]


numeric_missing = [
    'BsmtFinSF1', # median
    'BsmtFinSF2', # median

    'BsmtUnfSF', # median
    'GarageArea', # median
    
    'GarageYrBlt', # median
    'LotFrontage', # median
    'MasVnrArea', # median
    'TotalBsmtSF' # median
 ]

numeric_missing_most_frequent = [
    'GarageCars', # most frequent
    'BsmtFullBath', # most frequent
    'BsmtHalfBath', # most frequent
]

for col in zeroFlagList:
    X_numeric[f"{col}_flag"] = (X_numeric[col] == 0).astype(int)
    X_test[f"{col}_flag"] = (X_test[col] == 0).astype(int)

numeric_features = X_numeric.columns.tolist()  # Güncelle

for col in log1pList:
    X_numeric[col] = np.log1p(X_numeric[col])
    X_test[col] = np.log1p(X_test[col])


numeric_missing_pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

numeric_missing_most_frequent_pipe = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    StandardScaler()
)

te_pipe = make_pipeline(
    TargetEncoder(smooth="auto" , cv=5),
    StandardScaler()
)


# ordinal_data = pd.read_csv("./ordinal_features.csv")
# ordinal_features = ordinal_data.columns.tolist()
# for col in ordinal_data:
#     ordinal_data[col] = ordinal_data[col] + 1

# X_numeric = pd.concat([X_numeric, ordinal_data], axis=1)




preprocessor_numeric = make_column_transformer(
    ( numeric_missing_pipe , numeric_missing ),
    ( numeric_missing_most_frequent_pipe , numeric_missing_most_frequent ),
    ( te_pipe , targetEncodingList ),
    ( StandardScaler() ,  [col for col in numeric_features if col not in  numeric_missing + numeric_missing_most_frequent + targetEncodingList ] ),
    remainder="passthrough"

)

X_train, X_val, y_train, y_val = train_test_split(
    X_numeric, y, test_size=0.2, random_state=42
)


# def objective_elastic(trial):
#     params = {
#         "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
#         "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
#         "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
#         "max_iter": trial.suggest_int("max_iter", 100, 15000),
#     }

#     model_numeric = ElasticNet(**params, random_state=42)
#     pipeline_numeric = make_pipeline(
#     preprocessor_numeric,
#     model_numeric)

#     pipeline_numeric.fit(X_train, y_train)

#     preds_numeric = pipeline_numeric.predict(X_val)

#     score_total = root_mean_squared_error(y_val, preds_numeric)
#     return score_total


# study_elastic = optuna.create_study(direction="minimize")
# study_elastic.optimize(objective_elastic, n_trials=500)

# best_params_elastic = study_elastic.best_params
# print(best_params_elastic)

#######################################################

# def objective(trial):

#     params= {
#     "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]),
#     "metric": "rmse",
#     "objective": "regression",
#     #"force_row_wise": True,
#     "n_estimators": trial.suggest_int("n_estimators", 300, 15000),
#     "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5,log=True),
#     "num_leaves": trial.suggest_int("num_leaves", 10, 256),
#     "max_depth": trial.suggest_int("max_depth", 3, 8),
#     "min_child_samples": trial.suggest_int("min_child_samples", 10, 300,log=True),
#     "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
#     "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
#     "subsample": trial.suggest_float("subsample", 0.1, 1.0),
#     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0 , log=True),
#     "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0,log=True),
#     "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0,log=True),
#     "random_state": 42,
#     "verbosity": -1,

#     }



#     model = LGBMRegressor(**params)
    
#     pipeline = make_pipeline(
#     preprocessor_numeric,
#     model)

#     pipeline.fit(X_train, y_train)

#     preds = pipeline.predict(X_val)
#     score = root_mean_squared_error(y_val, preds)
    
#     return score

# study = optuna.create_study(
#     direction="minimize",
#     study_name="house_prices"
# )

# study.optimize(objective, n_trials=500)

# best_params = study.best_params
# print(best_params)

#######################################################

#0.13950289821861592
# best_params = {'boosting_type': 'gbdt', 'n_estimators': 8000, 'learning_rate': 0.002390189038056625, 'num_leaves': 44, 'max_depth': 3, 'min_child_samples': 10, 'min_child_weight': 0.002021735576790916, 'min_split_gain': 0.0009468802296035798, 'subsample': 0.1161456249264999, 'colsample_bytree': 0.6488062773218959, 'reg_alpha': 0.0654705459897717, 'reg_lambda': 0.09863245790562798}
# best_params.update({
#     "metric": "rmse",
#     "objective": "regression",
#     "random_state": 42,
#     "verbosity": -1,
# })

# best_model = LGBMRegressor(**best_params)
# best_pipeline = make_pipeline(
#     preprocessor_numeric,
#     best_model
# )

# cross_val_scores = cross_val_score(best_pipeline, X_numeric, y, cv=5, scoring="neg_root_mean_squared_error")
# cross_val_scores = -cross_val_scores

# print(cross_val_scores)
# print(cross_val_scores.mean())
# print(cross_val_scores.std())

#######################################################

def objective_total(trial):
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",  # veya "gpu_hist" (GPU varsa)
        "eval_metric": "rmse",
        "booster": trial.suggest_categorical("booster", ["gbtree"]), #, "dart"
        #'early_stopping_rounds': 100,
        
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

    model = XGBRegressor(**params)
    
    pipeline = make_pipeline(
    preprocessor_numeric,
    model)

    pipeline.fit(X_train, y_train)

    preds_total = pipeline.predict(X_val)

    score = root_mean_squared_error(y_val, preds_total)

    
    return score

# study_total = optuna.create_study(
#     direction="minimize",
#     study_name="house_prices_total"
# )

# study_total.optimize(objective_total, n_trials=500)

# best_params_total = study_total.best_params
# print(best_params_total)

#best_params = {'booster': 'gbtree', 'learning_rate': 0.015772622710502393, 'max_depth': 4, 'min_child_weight': 1.767521646696087, 'lambda': 0.10471403237877837, 'alpha': 0.0032944329415070615, 'subsample': 0.34168987238666015, 'colsample_bytree': 0.9607909164542489, 'rate_drop': 0.06458002231537555, 'skip_drop': 0.1399326025771156, 'n_estimators': 8171}
#0.13260348882385764 with ordinals

best_params = {'booster': 'gbtree', 'learning_rate': 0.01336963599195788, 'max_depth': 16, 'min_child_weight': 8.157405313496586, 'lambda': 0.15577441923833996, 'alpha': 0.025287573162511266, 'subsample': 0.2580340358790427, 'colsample_bytree': 0.4599431346725082, 'rate_drop': 0.23634615966636632, 'skip_drop': 0.18221036768379434, 'n_estimators': 714}
#.13298878398047012
best_params.update({
        "objective": "reg:squarederror",
        "tree_method": "hist",  # veya "gpu_hist" (GPU varsa)
        "eval_metric": "rmse",
        "random_state": 42,
        "verbosity": 0,
})



best_model = XGBRegressor(**best_params)
best_pipeline = make_pipeline(
    preprocessor_numeric,
    best_model
)

# cross_val_scores = cross_val_score(best_pipeline, X_numeric, y, cv=5, scoring="neg_root_mean_squared_error")
# cross_val_scores = -cross_val_scores

# print(cross_val_scores)
# print(cross_val_scores.mean())
# print(cross_val_scores.std())

skf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_numeric))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_numeric, y)):
    print(f"Fold {fold+1}")
    X_tr, X_val = X_numeric.iloc[tr_idx], X_numeric.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    best_pipeline.fit(X_tr, y_tr)

    oof_preds[val_idx] = best_pipeline.predict(X_val)
    test_preds += best_pipeline.predict(X_test) / skf.n_splits

    
meta_X = pd.DataFrame({"numeric_oof":oof_preds})
meta_test = pd.DataFrame({"numeric_oof_test":test_preds})

meta_X.to_csv("numeric_oof.csv",index=False)
meta_test.to_csv("numeric_oof_test.csv",index=False)