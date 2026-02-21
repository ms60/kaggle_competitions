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
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
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


#print(X.head())
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

X_categorical = X[categorical_features]
X_test = X[categorical_features]


missing_category_features = ["Alley","Fence","FireplaceQu","MasVnrType","MiscFeature","PoolQC","GarageCond","GarageFinish","GarageQual","GarageType"]
missing_imputation_features  = ["BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","BsmtQual","Electrical","Exterior1st","Exterior2nd","Functional","KitchenQual","MSZoning","SaleType","Utilities"]

X_categorical["Alley"] = X_categorical["Alley"].fillna("Missing")

X_categorical["BsmtCond"] = X_categorical["BsmtCond"].fillna("Missing")
X_categorical["BsmtCond"] = X_categorical["BsmtCond"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

X_categorical["BsmtExposure"] = X_categorical["BsmtExposure"].fillna("Missing")
X_categorical["BsmtExposure"] = X_categorical["BsmtExposure"].map({"Missing":0 , "No":1 , "Mn":2 , "Av":3 , "Gd":4})

X_categorical["BsmtFinType1"] = X_categorical["BsmtFinType1"].fillna("Missing")
X_categorical["BsmtFinType1"] = X_categorical["BsmtFinType1"].map({"Missing":0 , "Unf":1 , "LwQ":2 , "Rec":3 , "BLQ":4 , "ALQ":5 , "GLQ":6})

X_categorical["BsmtFinType2"] = X_categorical["BsmtFinType2"].fillna("Missing")
X_categorical["BsmtFinType2"] = X_categorical["BsmtFinType2"].map({"Missing":0 , "Unf":1 , "LwQ":2 , "Rec":3 , "BLQ":4 , "ALQ":5 , "GLQ":6})

X_categorical["BsmtQual"] = X_categorical["BsmtQual"].fillna("Missing")
X_categorical["BsmtQual"] = X_categorical["BsmtQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

X_categorical["Electrical"] = X_categorical["Electrical"].fillna("FuseF")
X_categorical["Electrical"] = X_categorical["Electrical"].map({"Missing":0 , "Mix":1 , "FuseP":2 , "FuseF":3 , "FuseA":4 , "SBrkr":5})

X_categorical["ExterCond"] = X_categorical["ExterCond"].map({ "Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})

X_categorical["ExterQual"] = X_categorical["ExterQual"].map({"Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})



X_categorical["Exterior1st"] =  X_categorical["Exterior1st"].fillna("Other")
X_categorical["Exterior2nd"] =  X_categorical["Exterior2nd"].fillna("Other")

X_categorical["Fence"] = X_categorical["Fence"].fillna("Missing")
X_categorical["Fence"] = X_categorical["Fence"].map({"Missing":0 , "MnWw":1 , "GdWo":2 , "MnPrv":3 , "GdPrv":4})

X_categorical["FireplaceQu"] = X_categorical["FireplaceQu"].fillna("Missing")
X_categorical["FireplaceQu"] = X_categorical["FireplaceQu"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})


X_categorical["Functional"] = X_categorical["Functional"].fillna("Typ")
X_categorical["Functional"] = X_categorical["Functional"].map({"Missing":0 , "Sal":1 , "Sev":2 , "Maj2":3 , "Maj1":4 , "Mod":5 , "Min2":6 , "Min1":7 , "Typ":8})

X_categorical["GarageCond"] = X_categorical["GarageCond"].fillna("Missing")
X_categorical["GarageCond"] = X_categorical["GarageCond"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

X_categorical["GarageFinish"] = X_categorical["GarageFinish"].fillna("Missing")
X_categorical["GarageFinish"] = X_categorical["GarageFinish"].map({"Missing":0 , "Unf":1 , "RFn":2 , "Fin":3})

X_categorical["GarageQual"] = X_categorical["GarageQual"].fillna("Missing")
X_categorical["GarageQual"] = X_categorical["GarageQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

X_categorical["GarageType"] = X_categorical["GarageType"].fillna("Missing")
X_categorical["GarageType"] = X_categorical["GarageType"].map({"Missing":0 , "Detchd":1 , "CarPort":2 , "BuiltIn":3 , "Basment":4 , "Attchd":5 , "2Types":6})

X_categorical["HeatingQC"] = X_categorical["HeatingQC"].map({"Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})


X_categorical["KitchenQual"] = X_categorical["KitchenQual"].fillna("TA")
X_categorical["KitchenQual"] = X_categorical["KitchenQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})


X_categorical["MSZoning"] = X_categorical["MSZoning"].fillna("RH")

X_categorical["MasVnrType"] = X_categorical["MasVnrType"].fillna("None")

X_categorical["MiscFeature"] = X_categorical["MiscFeature"].fillna("Missing")

X_categorical["PoolQC"] = X_categorical["PoolQC"].fillna("Missing")
X_categorical["PoolQC"] = X_categorical["PoolQC"].map({"Missing":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})

X_categorical["SaleType"] = X_categorical["SaleType"].fillna("Oth")

X_categorical["Utilities"] = X_categorical["Utilities"].fillna("AllPub")

X_categorical["HouseStyle"] = X_categorical["HouseStyle"].map({"1Story":1, "1.5Fin":2, "1.5Unf":3, "2Story":4, "2.5Fin":5, "2.5Unf":6, "SFoyer":7, "SLvl":8})

X_categorical["LandContour"] = X_categorical["LandContour"].map({"Lvl":1, "Bnk":2, "HLS":3 ,"Low":4})

X_categorical["LandSlope"] = X_categorical["LandSlope"].map({"Gtl":1, "Mod":2, "Sev":3})

X_categorical["LotShape"] = X_categorical["LotShape"].map({"Reg":1, "IR1":2, "IR2":3, "IR3":4})

X_categorical["PavedDrive"] = X_categorical["PavedDrive"].map({"Y":1, "P":2, "N":3})

X_categorical["SaleCondition"] = X_categorical["SaleCondition"].map({"Normal":1, "Abnorml":2, "AdjLand":3, "Alloca":4, "Family":5, "Partial":6})


#-----------------------------------

X_test["Alley"] = X_test["Alley"].fillna("Missing")

X_test["BsmtCond"] = X_test["BsmtCond"].fillna("Missing")
X_test["BsmtCond"] = X_test["BsmtCond"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

X_test["BsmtExposure"] = X_test["BsmtExposure"].fillna("Missing")
X_test["BsmtExposure"] = X_test["BsmtExposure"].map({"Missing":0 , "No":1 , "Mn":2 , "Av":3 , "Gd":4})

X_test["BsmtFinType1"] = X_test["BsmtFinType1"].fillna("Missing")
X_test["BsmtFinType1"] = X_test["BsmtFinType1"].map({"Missing":0 , "Unf":1 , "LwQ":2 , "Rec":3 , "BLQ":4 , "ALQ":5 , "GLQ":6})

X_test["BsmtFinType2"] = X_test["BsmtFinType2"].fillna("Missing")
X_test["BsmtFinType2"] = X_test["BsmtFinType2"].map({"Missing":0 , "Unf":1 , "LwQ":2 , "Rec":3 , "BLQ":4 , "ALQ":5 , "GLQ":6})

X_test["BsmtQual"] = X_test["BsmtQual"].fillna("Missing")
X_test["BsmtQual"] = X_test["BsmtQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

X_test["Electrical"] = X_test["Electrical"].fillna("FuseF")
X_test["Electrical"] = X_test["Electrical"].map({"Missing":0 , "Mix":1 , "FuseP":2 , "FuseF":3 , "FuseA":4 , "SBrkr":5})

X_test["ExterCond"] = X_test["ExterCond"].map({ "Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})

X_test["ExterQual"] = X_test["ExterQual"].map({"Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})



X_test["Exterior1st"] =  X_test["Exterior1st"].fillna("Other")
X_test["Exterior2nd"] =  X_test["Exterior2nd"].fillna("Other")

X_test["Fence"] = X_test["Fence"].fillna("Missing")
X_test["Fence"] = X_test["Fence"].map({"Missing":0 , "MnWw":1 , "GdWo":2 , "MnPrv":3 , "GdPrv":4})

X_test["FireplaceQu"] = X_test["FireplaceQu"].fillna("Missing")
X_test["FireplaceQu"] = X_test["FireplaceQu"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})


X_test["Functional"] = X_test["Functional"].fillna("Typ")
X_test["Functional"] = X_test["Functional"].map({"Missing":0 , "Sal":1 , "Sev":2 , "Maj2":3 , "Maj1":4 , "Mod":5 , "Min2":6 , "Min1":7 , "Typ":8})

X_test["GarageCond"] = X_test["GarageCond"].fillna("Missing")
X_test["GarageCond"] = X_test["GarageCond"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

X_test["GarageFinish"] = X_test["GarageFinish"].fillna("Missing")
X_test["GarageFinish"] = X_test["GarageFinish"].map({"Missing":0 , "Unf":1 , "RFn":2 , "Fin":3})

X_test["GarageQual"] = X_test["GarageQual"].fillna("Missing")
X_test["GarageQual"] = X_test["GarageQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

X_test["GarageType"] = X_test["GarageType"].fillna("Missing")
X_test["GarageType"] = X_test["GarageType"].map({"Missing":0 , "Detchd":1 , "CarPort":2 , "BuiltIn":3 , "Basment":4 , "Attchd":5 , "2Types":6})

X_test["HeatingQC"] = X_test["HeatingQC"].map({"Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})


X_test["KitchenQual"] = X_test["KitchenQual"].fillna("TA")
X_test["KitchenQual"] = X_test["KitchenQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})


X_test["MSZoning"] = X_test["MSZoning"].fillna("RH")

X_test["MasVnrType"] = X_test["MasVnrType"].fillna("None")

X_test["MiscFeature"] = X_test["MiscFeature"].fillna("Missing")

X_test["PoolQC"] = X_test["PoolQC"].fillna("Missing")
X_test["PoolQC"] = X_test["PoolQC"].map({"Missing":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})

X_test["SaleType"] = X_test["SaleType"].fillna("Oth")

X_test["Utilities"] = X_test["Utilities"].fillna("AllPub")

X_test["HouseStyle"] = X_test["HouseStyle"].map({"1Story":1, "1.5Fin":2, "1.5Unf":3, "2Story":4, "2.5Fin":5, "2.5Unf":6, "SFoyer":7, "SLvl":8})

X_test["LandContour"] = X_test["LandContour"].map({"Lvl":1, "Bnk":2, "HLS":3 ,"Low":4})

X_test["LandSlope"] = X_test["LandSlope"].map({"Gtl":1, "Mod":2, "Sev":3})

X_test["LotShape"] = X_test["LotShape"].map({"Reg":1, "IR1":2, "IR2":3, "IR3":4})

X_test["PavedDrive"] = X_test["PavedDrive"].map({"Y":1, "P":2, "N":3})

X_test["SaleCondition"] = X_test["SaleCondition"].map({"Normal":1, "Abnorml":2, "AdjLand":3, "Alloca":4, "Family":5, "Partial":6})

# print(X_categorical.head())
# print(X_categorical.isnull().sum())

binary_features = [
    col for col in X_categorical.columns
    if X_categorical[col].dropna().nunique() == 2 and set(X_categorical[col].dropna().unique()).issubset({0, 1})
]

ohe_features = X_categorical.select_dtypes(exclude=[np.number]).columns.tolist()
ordinal_features = [col for col in X_categorical.columns if col not in binary_features + ohe_features]

ordinal_features.remove("MoSold")
ordinal_features.remove("YrSold")

# ohe_features.append("MoSold")
# ohe_features.append("YrSold")


te_features = ["Exterior1st","Exterior2nd","MiscFeature","Neighborhood","RoofMatl","RoofStyle","SaleType","Utilities","MoSold","YrSold"]

te_pipeline = make_pipeline(
    TargetEncoder(smooth="auto" , cv=5),
    StandardScaler()
)

# for col in ordinal_features:
#     X_categorical[col] = X_categorical[col] + 1


preprocessor_categorical = make_column_transformer(
    ( OneHotEncoder(handle_unknown="ignore") , ohe_features ),
    ( StandardScaler() , ordinal_features ) , 
    ( te_pipeline , te_features ),
    remainder="drop"
)

X_train, X_val, y_train, y_val = train_test_split(
    X_categorical, y, test_size=0.2, random_state=42
)



# def objective_total(trial):
#     params = {
#         "objective": "reg:squarederror",
#         "tree_method": "hist",  # veya "gpu_hist" (GPU varsa)
#         "eval_metric": "rmse",
#         "booster": trial.suggest_categorical("booster", ["gbtree"]), #, "dart"
#         #'early_stopping_rounds': 100,
        
#         # Öğrenme oranı ve derinlik
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.9, log=True),
#         "max_depth": trial.suggest_int("max_depth", 3, 32),
#         "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        
#         # Düzenlileştirme
#         "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),   # L2
#         "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),     # L1
        
#         # Alt örnekleme
#         "subsample": trial.suggest_float("subsample", 0.1, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        
#         # DART için özel dropout parametreleri (booster=dart olduğunda aktif)
#         "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.5),
#         "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.5),
        
#         # Ağaç sayısı
#         "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
#         "n_jobs": -1,
#         "random_state": 42,
        
#     }

#     model = XGBRegressor(**params)
    
#     pipeline = make_pipeline(
#     preprocessor_categorical,
#     model)

#     pipeline.fit(X_train, y_train)

#     preds_total = pipeline.predict(X_val)

#     score = root_mean_squared_error(y_val, preds_total)

    
#     return score

# study_total = optuna.create_study(
#     direction="minimize",
#     study_name="house_prices_total"
# )

# study_total.optimize(objective_total, n_trials=500)

# best_params = study_total.best_params
# print(best_params)

best_params = {'booster': 'gbtree', 'learning_rate': 0.011839030674207798, 'max_depth': 3, 'min_child_weight': 1.7728203974529972, 'lambda': 0.43067221413674694, 'alpha': 0.0022261933720581647, 'subsample': 0.6169568800381058, 'colsample_bytree': 0.42432845291859156, 'rate_drop': 0.47334693664333655, 'skip_drop': 0.09287668394760126, 'n_estimators': 3528}
# 0.1655754483303403
# 0.011472365715120206

#best_params = {'booster': 'gbtree', 'learning_rate': 0.01492195436579292, 'max_depth': 3, 'min_child_weight': 1.8306287694479342, 'lambda': 0.002813700569873711, 'alpha': 0.01597303798568678, 'subsample': 0.8505213425709277, 'colsample_bytree': 0.9667067014967649, 'rate_drop': 0.2912204501343599, 'skip_drop': 0.4387250930988215, 'n_estimators': 2675}
# 0.16749671053250387
# 0.011077892612727385

best_params.update({
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,

})

best_model = XGBRegressor(**best_params)
best_pipeline = make_pipeline(
    preprocessor_categorical,
    best_model
)

# cross_val_scores = cross_val_score(best_pipeline, X_categorical, y, cv=5, scoring="neg_root_mean_squared_error")
# cross_val_scores = -cross_val_scores

# print(cross_val_scores)
# print(cross_val_scores.mean())
# print(cross_val_scores.std())



skf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_categorical))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_categorical, y)):
    print(f"Fold {fold+1}")
    X_tr, X_val = X_categorical.iloc[tr_idx], X_categorical.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    best_pipeline.fit(X_tr, y_tr)

    oof_preds[val_idx] = best_pipeline.predict(X_val)
    test_preds += best_pipeline.predict(X_test) / skf.n_splits

    
meta_X = pd.DataFrame({"categorical_oof":oof_preds})
meta_test = pd.DataFrame({"categorical_oof_test":test_preds})

meta_X.to_csv("categorical_oof.csv",index=False)
meta_test.to_csv("categorical_oof_test.csv",index=False)


#X_categorical[ordinal_features].to_csv("ordinal_features.csv", index=False)


