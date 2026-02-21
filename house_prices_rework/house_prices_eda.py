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
from sklearn.model_selection import train_test_split
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


X_test = test.drop("Id",axis=1)

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




# EDA

# 1. Target analysis

def target_analysis(y):
    fig,axes = plt.subplots(3,3,figsize=(18, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    sns.histplot(y, bins=50, ax=axes[0,0], kde=False)
    axes[0,0].set_title("Target Histogram")

    sns.kdeplot(y, ax=axes[0,1])
    axes[0,1].set_title("Target Density Plot")

    sns.boxplot(y, orient='h', ax=axes[0,2])
    axes[0,2].set_title("Target Box Plot")

    stats.probplot(y,plot=axes[1,0])
    axes[1,0].set_title("Target Q-Q Plot")

    fig.tight_layout(pad=2.0)
    plt.show()

print(y.describe().T)

#target_analysis( y)

#target analysis conclusion
# target is skewed
# there are outlier in target

# target_analysis( y[y>300000] )

# 2. Univariate analysis

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()


def univariate_analysis_numeric(X, features, size=3):
    num_features = len(features)
    num_cols = min(size, num_features)
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 12))
    
    if num_features == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for idx, feature in enumerate(features):
        
        series = X[feature]
        missing_count = series.isna().sum()
        missing_ratio = missing_count / len(series) * 100

        sns.histplot(
            data=X,
            x=feature,
            bins=40,
            kde=True,
            ax=axes[idx]
        )

        # Başlık yerine xlabel'e yazıyoruz
        axes[idx].set_xlabel(
            f"{feature} | Missing: {missing_count} ({missing_ratio:.2f}%)",
            fontsize=8
        )

        axes[idx].set_title("")  # title boş



    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(pad=1.0)
    plt.show()

#univariate_analysis_numeric( X,numeric_features,6 )



# numeric univariate conclusion
# 1stFlrSF is numeric and skewed
# 2ndFlrSF numeric , lots of value at 0 , check again
# 3SsnPorch numeric , lots of value at 0 , too rare values at middle , can be rare encoding , check again
# BedroomAbvGr , numeric less then 10 , can be ordinal or nominal category , check again
# BsmtFinSF1 , numeric lots of value at 0
# BsmtFinSF2,numeric lots of value at 0 , too rare values at middle , can be rare encoding , check again
# BsmtFullBath numeric less then 10 , can be ordinal or nominal category , check again
# BsmtHalfBath numeric less then 10 , can be ordinal or nominal category , check again
# BsmtUnfSF numeric , lots of value at 0 , check again
# EnclosedPorch numeric lots of value at 0 , too rare values at middle , can be rare encoding , check again
# FirePlaces numeric
# FullBath numeric
# GarageArea numeric
# GarageCars numeric
# GarageYearBuilt , convert to years old
# GrLivArea , numeric , skewed
# HalfBath numeric
# KitchenAbvGr , can be removed 
# LotArea numeric skewed
# LotFrontage numeric skewed
# LowQualFinSF , numeric , lots of value at 0 , too rare values at middle
# MsSubClass , can be categorical
# MasVnrArea , numeric , skewed , lots of value at 0
# MiscVal , numeric , lots of value at 0 , too rare values at middle
# MoSold , numeric
# OpenPorchSF , numeric , skewed,
# OverallCond , can be ordinal
# OverallQual , can be ordinal
# PoolArea ,  numeric , lots of value at 0 , too rare values at middle
# ScreenPorch ,  numeric , lots of value at 0 , too rare values at middle
# TotRmsAbvGrd , normal
# TotalBsmtSF , slightly skewed , outliers 
# WoodDeckSF , numeric , lots of value at 0 , too rare values at middle
# YearBuilt , convert to years old
# YearRemodAdd , convert to years old
# YrsOld 

def univariate_analysis_categorical(X, features, size=3):
    num_features = len(features)
    num_cols = min(size, num_features)
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 12))
    
    
    if num_features == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for idx, feature in enumerate(features):
        
        # ORİJİNAL VERİYE DOKUNMUYORUZ
        temp_series = X[feature].copy()
        temp_series = temp_series.fillna("Missing")

        order = temp_series.value_counts().index[:20]

        sns.countplot(
            x=temp_series,
            ax=axes[idx],
            order=order
        )

        axes[idx].tick_params(axis='x', rotation=45)
        #axes[idx].set_title(f"{feature}", fontsize=11)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(pad=1.0)
    plt.show()

# univariate_analysis_categorical( X,categorical_features[:14],7 )
# univariate_analysis_categorical( X,categorical_features[14:28],7 )
# univariate_analysis_categorical( X,categorical_features[28:],7 )


def bivariate_analysis_numeric(X, features, y, size=3):
    
    num_features = len(features)
    num_cols = min(size, num_features)
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, 
                             ncols=num_cols, 
                             figsize=(18, 12))

    # Eğer tek satır/tek kolon ise flatten sorunu çöz
    if num_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(features):
        
        ax = axes[i]

        # Scatter
        sns.scatterplot(x=X[col], y=y, alpha=0.3, ax=ax)

        # Regression line
        sns.regplot(x=X[col], y=y,
                    scatter=False,
                    lowess=False,
                    line_kws={"color": "red"},
                    ax=ax)

        # LOWESS (nonlinear trend)
        sns.regplot(x=X[col], y=y,
                    scatter=False,
                    lowess=True,
                    line_kws={"color": "green"},
                    ax=ax)

        # Correlation hesapla
        corr, _ = pearsonr(X[col].fillna(0), y.fillna(0))

        ax.set_title(f"Pearson r = {corr:.3f}")
        
        # # Border ekleyelim (daha okunabilir olsun)
        # for spine in ax.spines.values():
        #     spine.set_edgecolor("gray")
        #     spine.set_linewidth(1)

    # Fazla boş subplot varsa kapatalım
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

bivariate_analysis_numeric(X, numeric_features, y, size=6)
# categorical univariate conclusion


def bivariate_analysis_categorical(X, features, y, size=2, top_n=10):
    
    df = X.copy()
    df["_target_"] = y
    
    num_features = len(features)
    num_cols = min(size, num_features)
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, 
                             ncols=num_cols, 
                             figsize=(18, 12))

    if num_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(features):

        ax = axes[i]

        # Çok fazla kategori varsa top_n al
        top_categories = df[col].value_counts().nlargest(top_n).index
        temp = df[df[col].isin(top_categories)]

        # Grup ortalamaları
        group_means = temp.groupby(col)["_target_"].mean().sort_values()

        # ANOVA için gruplar
        groups = [temp[temp[col] == cat]["_target_"].dropna()
                  for cat in group_means.index]

        if len(groups) > 1:
            f_stat, p_value = f_oneway(*groups)
        else:
            p_value = np.nan

        # Boxplot
        sns.boxplot(x=col, y="_target_", data=temp,
                    order=group_means.index,
                    ax=ax)

        ax.set_title(f"{col}\nANOVA p = {p_value:.4f}")
        ax.tick_params(axis='x', rotation=45)

        # # Border
        # for spine in ax.spines.values():
        #     spine.set_edgecolor("gray")
        #     spine.set_linewidth(1)

    # Fazla subplotları kapat
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    # Mean table döndür
    for col in features:
        print(f"\n===== {col} =====")
        print(df.groupby(col)["_target_"].agg(["count","mean","std"]).sort_values("mean"))

    
#bivariate_analysis_categorical(X,categorical_features,y,6)

# numeric_features.remove("MoSold")
# numeric_features.remove("YrSold")

# categorical_features.append("MoSold")
# categorical_features.append("YrSold")


# X_numeric = X[numeric_features]
# X_categorical = X[categorical_features]

# zeroFlagList = [  "2ndFlrSF" , "3SsnPorch" , "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","EnclosedPorch","GarageArea","LowQualFinSF","MSSubClass","MasVnrArea","MiscVal","OpenPorchSF","PoolArea","ScreenPorch","WoodDeckSF","YearRemodAdd"]

# log1pList = [ "1stFlrSF","2ndFlrSF","3SsnPorch","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF" , "EnclosedPorch","GarageArea","GarageYrBlt","GrLivArea","LotArea","LotFrontage","MasVnrArea","MiscVal","OpenPorchSF",
#  "PoolArea","ScreenPorch","TotalBsmtSF","WoodDeckSF","YearBuilt","YearRemodAdd"]

# targetEncodingList = ["BedroomAbvGr","Fireplaces","FullBath","GarageCars","HalfBath","KitchenAbvGr","LowQualFinSF","MSSubClass","OverallCond","OverallQual","TotRmsAbvGrd"]

# toCategoricalList = ["MoSold","YrSold"]



# # print("-"*80)
# # print( [col for col in log1pList if col not in numeric_features] )
# # print( [col for col in targetEncodingList if col not in numeric_features] )


# numeric_missing = [
#     'BsmtFinSF1', # median
#     'BsmtFinSF2', # median

#     'BsmtUnfSF', # median
#     'GarageArea', # median
    
#     'GarageYrBlt', # median
#     'LotFrontage', # median
#     'MasVnrArea', # median
#     'TotalBsmtSF' # median
#  ]

# numeric_missing_most_frequent = [
#     'GarageCars', # most frequent
#     'BsmtFullBath', # most frequent
#     'BsmtHalfBath', # most frequent
# ]

# for col in zeroFlagList:
#     X_numeric[f"{col}_flag"] = (X_numeric[col] == 0).astype(int)

# te_pipeline = make_pipeline(
#     TargetEncoder(smooth="auto", cv=3),
#     StandardScaler()
# )    

# preprocessor = make_column_transformer(
#     ( SimpleImputer(strategy='median') ,numeric_missing ),
#     ( SimpleImputer(strategy='most_frequent') ,numeric_missing_most_frequent ),
#     ( FunctionTransformer(lambda X: np.log1p(X)) , log1pList  ),
#     #( TargetEncoder(smooth="auto" , cv=5) , targetEncodingList ),
#     ( StandardScaler() , numeric_features ),
#     remainder="passthrough"

# )

# X_train, X_val, y_train, y_val = train_test_split(
#     X_numeric, y, test_size=0.2, random_state=42
# )

# # model = LGBMRegressor(
# #     n_estimators=1000,
# #     learning_rate=0.05,
# #     max_depth=-1,
# #     random_state=42
# # )


# # pipeline = make_pipeline(
# #     preprocessor,
# #     model)

# # pipeline.fit(X_train, y_train)

# # preds = pipeline.predict(X_val)


# # score = root_mean_squared_error(y_val, preds)
# # print(score)

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
#     preprocessor,
#     model)

#     pipeline.fit(X_train, y_train)

#     preds = pipeline.predict(X_val)
#     score = root_mean_squared_error(y_val, preds)
    
#     return score

# # study = optuna.create_study(
# #     direction="minimize",
# #     study_name="house_prices"
# # )

# # study.optimize(objective, n_trials=500)

# # best_params = study.best_params
# # print(best_params)

# # row_wise {'boosting_type': 'gbdt', 'n_estimators': 10443, 'learning_rate': 0.0022795934862992113, 'num_leaves': 166, 'max_depth': 3, 'min_child_samples': 11, 'min_child_weight': 0.15337151195996931, 'min_split_gain': 0.03645087073359092, 'subsample': 0.22422477740942026, 'colsample_bytree': 0.25801505879546993, 'reg_alpha': 0.0034368324550785737, 'reg_lambda': 0.0024639123286230758}

# #{'boosting_type': 'gbdt', 'n_estimators': 688, 'learning_rate': 0.05510650672460545, 'num_leaves': 226, 'max_depth': 3, 'min_child_samples': 12, 'min_child_weight': 0.01620193547575798, 'min_split_gain': 0.04030074967273144, 'subsample': 0.91840480010437, 'colsample_bytree': 0.3279876712817882, 'reg_alpha': 0.110248185330013, 'reg_lambda': 0.10916554627217187}
# # 0.13546303861844716

# #without te
# # {'boosting_type': 'gbdt', 'n_estimators': 4132, 'learning_rate': 0.02191445501318898, 'num_leaves': 194, 'max_depth': 3, 'min_child_samples': 11, 'min_child_weight': 0.1866715963349142, 'min_split_gain': 0.01803978746830913, 'subsample': 0.6546231224407715, 'colsample_bytree': 0.323783898612738, 'reg_alpha': 0.010940179937347519, 'reg_lambda': 0.006414671996452862}
# #  0.13506163098992607

# missing_category_features = ["Alley","Fence","FireplaceQu","MasVnrType","MiscFeature","PoolQC","GarageCond","GarageFinish","GarageQual","GarageType"]
# missing_imputation_features  = ["BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","BsmtQual","Electrical","Exterior1st","Exterior2nd","Functional","KitchenQual","MSZoning","SaleType","Utilities"]

# X_categorical["Alley"] = X_categorical["Alley"].fillna("Missing")

# X_categorical["BsmtCond"] = X_categorical["BsmtCond"].fillna("Missing")
# X_categorical["BsmtCond"] = X_categorical["BsmtCond"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

# X_categorical["BsmtExposure"] = X_categorical["BsmtExposure"].fillna("Missing")
# X_categorical["BsmtExposure"] = X_categorical["BsmtExposure"].map({"Missing":0 , "No":1 , "Mn":2 , "Av":3 , "Gd":4})

# X_categorical["BsmtFinType1"] = X_categorical["BsmtFinType1"].fillna("Missing")
# X_categorical["BsmtFinType1"] = X_categorical["BsmtFinType1"].map({"Missing":0 , "Unf":1 , "LwQ":2 , "Rec":3 , "BLQ":4 , "ALQ":5 , "GLQ":6})

# X_categorical["BsmtFinType2"] = X_categorical["BsmtFinType2"].fillna("Missing")
# X_categorical["BsmtFinType2"] = X_categorical["BsmtFinType2"].map({"Missing":0 , "Unf":1 , "LwQ":2 , "Rec":3 , "BLQ":4 , "ALQ":5 , "GLQ":6})

# X_categorical["BsmtQual"] = X_categorical["BsmtQual"].fillna("Missing")
# X_categorical["BsmtQual"] = X_categorical["BsmtQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

# X_categorical["Electrical"] = X_categorical["Electrical"].fillna("FuseF")
# X_categorical["Electrical"] = X_categorical["Electrical"].map({"Missing":0 , "Mix":1 , "FuseP":2 , "FuseF":3 , "FuseA":4 , "SBrkr":5})

# X_categorical["ExterCond"] = X_categorical["ExterCond"].map({ "Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})

# X_categorical["ExterQual"] = X_categorical["ExterQual"].map({"Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})



# X_categorical["Exterior1st"] =  X_categorical["Exterior1st"].fillna("Other")
# X_categorical["Exterior2nd"] =  X_categorical["Exterior2nd"].fillna("Other")

# X_categorical["Fence"] = X_categorical["Fence"].fillna("Missing")
# X_categorical["Fence"] = X_categorical["Fence"].map({"Missing":0 , "MnWw":1 , "GdWo":2 , "MnPrv":3 , "GdPrv":4})

# X_categorical["FireplaceQu"] = X_categorical["FireplaceQu"].fillna("Missing")
# X_categorical["FireplaceQu"] = X_categorical["FireplaceQu"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})


# X_categorical["Functional"] = X_categorical["Functional"].fillna("Typ")
# X_categorical["Functional"] = X_categorical["Functional"].map({"Missing":0 , "Sal":1 , "Sev":2 , "Maj2":3 , "Maj1":4 , "Mod":5 , "Min2":6 , "Min1":7 , "Typ":8})

# X_categorical["GarageCond"] = X_categorical["GarageCond"].fillna("Missing")
# X_categorical["GarageCond"] = X_categorical["GarageCond"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

# X_categorical["GarageFinish"] = X_categorical["GarageFinish"].fillna("Missing")
# X_categorical["GarageFinish"] = X_categorical["GarageFinish"].map({"Missing":0 , "Unf":1 , "RFn":2 , "Fin":3})

# X_categorical["GarageQual"] = X_categorical["GarageQual"].fillna("Missing")
# X_categorical["GarageQual"] = X_categorical["GarageQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

# X_categorical["GarageType"] = X_categorical["GarageType"].fillna("Missing")
# X_categorical["GarageType"] = X_categorical["GarageType"].map({"Missing":0 , "Detchd":1 , "CarPort":2 , "BuiltIn":3 , "Basment":4 , "Attchd":5 , "2Types":6})

# X_categorical["HeatingQC"] = X_categorical["HeatingQC"].map({"Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})


# X_categorical["KitchenQual"] = X_categorical["KitchenQual"].fillna("TA")
# X_categorical["KitchenQual"] = X_categorical["KitchenQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})


# X_categorical["MSZoning"] = X_categorical["MSZoning"].fillna("RH")

# X_categorical["MasVnrType"] = X_categorical["MasVnrType"].fillna("None")

# X_categorical["MiscFeature"] = X_categorical["MiscFeature"].fillna("Missing")

# X_categorical["PoolQC"] = X_categorical["PoolQC"].fillna("Missing")
# X_categorical["PoolQC"] = X_categorical["PoolQC"].map({"Missing":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})

# X_categorical["SaleType"] = X_categorical["SaleType"].fillna("Oth")

# X_categorical["Utilities"] = X_categorical["Utilities"].fillna("AllPub")

# X_categorical["HouseStyle"] = X_categorical["HouseStyle"].map({"1Story":1, "1.5Fin":2, "1.5Unf":3, "2Story":4, "2.5Fin":5, "2.5Unf":6, "SFoyer":7, "SLvl":8})

# X_categorical["LandContour"] = X_categorical["LandContour"].map({"Lvl":1, "Bnk":2, "HLS":3 ,"Low":4})

# X_categorical["LandSlope"] = X_categorical["LandSlope"].map({"Gtl":1, "Mod":2, "Sev":3})

# X_categorical["LotShape"] = X_categorical["LotShape"].map({"Reg":1, "IR1":2, "IR2":3, "IR3":4})

# X_categorical["PavedDrive"] = X_categorical["PavedDrive"].map({"Y":1, "P":2, "N":3})

# X_categorical["SaleCondition"] = X_categorical["SaleCondition"].map({"Normal":1, "Abnorml":2, "AdjLand":3, "Alloca":4, "Family":5, "Partial":6})






# ohe_features = ["Alley","BldgType","CentralAir","Condition1","Condition2","Foundation","Heating" , "LotConfig","MSZoning","MasVnrType","Street"]
# te_features = ["Exterior1st","Exterior2nd","MiscFeature","Neighborhood","RoofMatl","RoofStyle","SaleType","Utilities","MoSold","YrSold"]
# ordinal_features = [col for col in categorical_features if col not in ohe_features+te_features]



# # TargetEncoder + Scaler pipeline'ı
# # te_pipeline = make_pipeline(
# #     TargetEncoder(smooth="auto", cv=3),
# #     StandardScaler()
# # )

# preproces_categorical = make_column_transformer(
#     (StandardScaler(),ordinal_features),
#     (OneHotEncoder() , ohe_features),  
#     #( TargetEncoder(smooth="auto",cv=5) , te_features ),
#     (te_pipeline, te_features + targetEncodingList ),
#     remainder="drop"
# )

# # model_cat = LGBMRegressor(
# #     n_estimators=1000,
# #     learning_rate=0.05,
# #     max_depth=-1,
# #     random_state=42
# # )

# # X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(
# #     X_categorical, y, test_size=0.2, random_state=42
# # )

# # pipeline_cat = make_pipeline(
# #     preproces_categorical,
# #     model_cat)

# # pipeline_cat.fit(X_train_cat, y_train_cat)

# # preds_cat = pipeline_cat.predict(X_val_cat)


# # score_cat = root_mean_squared_error(y_val_cat, preds_cat)
# # print(score_cat)

#     # ( SimpleImputer(strategy='median') ,numeric_missing ),
#     # ( SimpleImputer(strategy='most_frequent') ,numeric_missing_most_frequent ),
#     # ( FunctionTransformer(lambda X: np.log1p(X)) , log1pList  ),
#     # #( TargetEncoder(smooth="auto" , cv=5) , targetEncodingList ),
#     # ( StandardScaler() , numeric_features ),
#     # remainder="passthrough"

#     # (StandardScaler(),ordinal_features),
#     # (OneHotEncoder() , ohe_features),  
#     # #( TargetEncoder(smooth="auto",cv=5) , te_features ),
#     # (te_pipeline, te_features),
#     # remainder="drop"


# # log1p_pipeline = make_pipeline(
# #     SimpleImputer(strategy="median"),
# #     FunctionTransformer(np.log1p)
# # )

# # numeric_pipeline = make_pipeline(
# #     SimpleImputer(strategy="median"),
# #     StandardScaler()
# # )

# # preprocess_total = make_column_transformer(
# #     (log1p_pipeline, log1pList),
# #     (SimpleImputer(strategy='most_frequent'), numeric_missing_most_frequent),
# #     (numeric_pipeline, numeric_features + ordinal_features),
# #     (OneHotEncoder(), ohe_features), 
# #     (te_pipeline, te_features + targetEncodingList),
# #     remainder="passthrough"
# # )

# X_total = pd.concat([X_numeric,X_categorical],axis=1)

# X_train_total, X_val_total, y_train_total, y_val_total = train_test_split(
#     X_total, y, test_size=0.2, random_state=42
# )

# imp_dict_numeric_missing = X_train_total.median().to_dict()
# imp_numeric_missing_most_frequent = X_train_total.mode().iloc[0].to_dict()

# X_train_total.fillna(imp_dict_numeric_missing, inplace=True)
# X_val_total.fillna(imp_dict_numeric_missing, inplace=True)

# X_train_total.fillna(imp_numeric_missing_most_frequent, inplace=True)
# X_val_total.fillna(imp_numeric_missing_most_frequent, inplace=True)

# X_test.fillna(imp_dict_numeric_missing, inplace=True)
# X_test.fillna(imp_numeric_missing_most_frequent, inplace=True)

# for col in log1pList:
#     X_train_total[col] = np.log1p(X_train_total[col])
#     X_val_total[col] = np.log1p(X_val_total[col])
#     X_test[col] = np.log1p(X_test[col])




# preprocess_total = make_column_transformer(
#     # ( SimpleImputer(strategy='median') ,numeric_missing ),
#     # ( SimpleImputer(strategy='most_frequent') ,numeric_missing_most_frequent ),
#     ( FunctionTransformer(lambda X: np.log1p(X)) , log1pList ),
#     ( StandardScaler() , numeric_features + ordinal_features ),
#     (OneHotEncoder(handle_unknown="ignore") , ohe_features),
#     (te_pipeline, te_features + targetEncodingList ),
#     remainder="passthrough" 
# )








# print(X_categorical.head())
# print("----------------")
# print(X_total.head())
# print(X_total.shape)







# def objective_total(trial):

#     # params= {
#     # "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]),
#     # "metric": "rmse",
#     # "objective": "regression",
#     # #"force_row_wise": True,
#     # "n_estimators": trial.suggest_int("n_estimators", 300, 15000),
#     # "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5,log=True),
#     # "num_leaves": trial.suggest_int("num_leaves", 10, 256),
#     # "max_depth": trial.suggest_int("max_depth", 3, 8),
#     # "min_child_samples": trial.suggest_int("min_child_samples", 10, 300,log=True),
#     # "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
#     # "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
#     # "subsample": trial.suggest_float("subsample", 0.1, 1.0),
#     # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0 , log=True),
#     # "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0,log=True),
#     # "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0,log=True),
#     # "random_state": 42,
#     # "verbosity": -1,

#     # }



#     # model_total = LGBMRegressor(**params)

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

#     model_total = XGBRegressor(**params)
    
#     pipeline_total = make_pipeline(
#     preprocess_total,
#     model_total)

#     pipeline_total.fit(X_train_total, y_train_total)

#     preds_total = pipeline_total.predict(X_val_total)

#     score_total = root_mean_squared_error(y_val_total, preds_total)

    
#     return score_total

# # study_total = optuna.create_study(
# #     direction="minimize",
# #     study_name="house_prices_total"
# # )

# # study_total.optimize(objective_total, n_trials=500)

# # best_params_total = study_total.best_params
# # print(best_params_total)

# # with te_cat
# # {'boosting_type': 'gbdt', 'n_estimators': 8798, 'learning_rate': 0.06731673948621261, 'num_leaves': 10, 'max_depth': 5, 'min_child_samples': 22, 'min_child_weight': 0.19474946934502294, 'min_split_gain': 0.017192693536087957, 'subsample': 0.5240347136409134, 'colsample_bytree': 0.23592946773416779, 'reg_alpha': 0.0015319939240991925, 'reg_lambda': 0.006962608238256599}
# # 0.13103122063182607

# # with te both
# #{'boosting_type': 'gbdt', 'n_estimators': 3401, 'learning_rate': 0.012403203472940544, 'num_leaves': 186, 'max_depth': 3, 'min_child_samples': 45, 'min_child_weight': 0.009180467571691761, 'min_split_gain': 0.0017618095790247808, 'subsample': 0.8861764184266921, 'colsample_bytree': 0.2741184896096117, 'reg_alpha': 0.055913323561942826, 'reg_lambda': 0.05581010414751861}
# #0.1290984518470964.

# #------------------------------------------------------
# # elasticnet model

# print(X_train_total.isnull().sum().sum())

# for col in X_train_total.columns:
#     if X_train_total[col].isnull().sum() > 0:
#         print(col)


# def objective_elastic(trial):
#     params = {
#         "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
#         "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
#         "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
#         "max_iter": trial.suggest_int("max_iter", 100, 15000),
#     }

#     model_total = ElasticNet(**params, random_state=42)
#     pipeline_total = make_pipeline(
#     preprocess_total,
#     model_total)

#     pipeline_total.fit(X_train_total, y_train_total)

#     preds_total = pipeline_total.predict(X_val_total)

#     score_total = root_mean_squared_error(y_val_total, preds_total)
#     return score_total


# study_elastic = optuna.create_study(direction="minimize")
# study_elastic.optimize(objective_elastic, n_trials=500)

# best_params_elastic = study_elastic.best_params
# print(best_params_elastic)

# #---------------------------------------------------------


# X_test["Alley"] = X_test["Alley"].fillna("Missing")

# X_test["BsmtCond"] = X_test["BsmtCond"].fillna("Missing")
# X_test["BsmtCond"] = X_test["BsmtCond"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

# X_test["BsmtExposure"] = X_test["BsmtExposure"].fillna("Missing")
# X_test["BsmtExposure"] = X_test["BsmtExposure"].map({"Missing":0 , "No":1 , "Mn":2 , "Av":3 , "Gd":4})

# X_test["BsmtFinType1"] = X_test["BsmtFinType1"].fillna("Missing")
# X_test["BsmtFinType1"] = X_test["BsmtFinType1"].map({"Missing":0 , "Unf":1 , "LwQ":2 , "Rec":3 , "BLQ":4 , "ALQ":5 , "GLQ":6})

# X_test["BsmtFinType2"] = X_test["BsmtFinType2"].fillna("Missing")
# X_test["BsmtFinType2"] = X_test["BsmtFinType2"].map({"Missing":0 , "Unf":1 , "LwQ":2 , "Rec":3 , "BLQ":4 , "ALQ":5 , "GLQ":6})

# X_test["BsmtQual"] = X_test["BsmtQual"].fillna("Missing")
# X_test["BsmtQual"] = X_test["BsmtQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

# X_test["Electrical"] = X_test["Electrical"].fillna("FuseF")
# X_test["Electrical"] = X_test["Electrical"].map({"Missing":0 , "Mix":1 , "FuseP":2 , "FuseF":3 , "FuseA":4 , "SBrkr":5})

# X_test["ExterCond"] = X_test["ExterCond"].map({ "Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})

# X_test["ExterQual"] = X_test["ExterQual"].map({"Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})



# X_test["Exterior1st"] =  X_test["Exterior1st"].fillna("Other")
# X_test["Exterior2nd"] =  X_test["Exterior2nd"].fillna("Other")

# X_test["Fence"] = X_test["Fence"].fillna("Missing")
# X_test["Fence"] = X_test["Fence"].map({"Missing":0 , "MnWw":1 , "GdWo":2 , "MnPrv":3 , "GdPrv":4})

# X_test["FireplaceQu"] = X_test["FireplaceQu"].fillna("Missing")
# X_test["FireplaceQu"] = X_test["FireplaceQu"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})


# X_test["Functional"] = X_test["Functional"].fillna("Typ")
# X_test["Functional"] = X_test["Functional"].map({"Missing":0 , "Sal":1 , "Sev":2 , "Maj2":3 , "Maj1":4 , "Mod":5 , "Min2":6 , "Min1":7 , "Typ":8})

# X_test["GarageCond"] = X_test["GarageCond"].fillna("Missing")
# X_test["GarageCond"] = X_test["GarageCond"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

# X_test["GarageFinish"] = X_test["GarageFinish"].fillna("Missing")
# X_test["GarageFinish"] = X_test["GarageFinish"].map({"Missing":0 , "Unf":1 , "RFn":2 , "Fin":3})

# X_test["GarageQual"] = X_test["GarageQual"].fillna("Missing")
# X_test["GarageQual"] = X_test["GarageQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})

# X_test["GarageType"] = X_test["GarageType"].fillna("Missing")
# X_test["GarageType"] = X_test["GarageType"].map({"Missing":0 , "Detchd":1 , "CarPort":2 , "BuiltIn":3 , "Basment":4 , "Attchd":5 , "2Types":6})

# X_test["HeatingQC"] = X_test["HeatingQC"].map({"Po":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})


# X_test["KitchenQual"] = X_test["KitchenQual"].fillna("TA")
# X_test["KitchenQual"] = X_test["KitchenQual"].map({"Missing":0 , "Po":1 , "Fa":2 , "TA":3 , "Gd":4 , "Ex":5})


# X_test["MSZoning"] = X_test["MSZoning"].fillna("RH")

# X_test["MasVnrType"] = X_test["MasVnrType"].fillna("None")

# X_test["MiscFeature"] = X_test["MiscFeature"].fillna("Missing")

# X_test["PoolQC"] = X_test["PoolQC"].fillna("Missing")
# X_test["PoolQC"] = X_test["PoolQC"].map({"Missing":0 , "Fa":1 , "TA":2 , "Gd":3 , "Ex":4})

# X_test["SaleType"] = X_test["SaleType"].fillna("Oth")

# X_test["Utilities"] = X_test["Utilities"].fillna("AllPub")

# X_test["HouseStyle"] = X_test["HouseStyle"].map({"1Story":1, "1.5Fin":2, "1.5Unf":3, "2Story":4, "2.5Fin":5, "2.5Unf":6, "SFoyer":7, "SLvl":8})

# X_test["LandContour"] = X_test["LandContour"].map({"Lvl":1, "Bnk":2, "HLS":3 ,"Low":4})

# X_test["LandSlope"] = X_test["LandSlope"].map({"Gtl":1, "Mod":2, "Sev":3})

# X_test["LotShape"] = X_test["LotShape"].map({"Reg":1, "IR1":2, "IR2":3, "IR3":4})

# X_test["PavedDrive"] = X_test["PavedDrive"].map({"Y":1, "P":2, "N":3})

# X_test["SaleCondition"] = X_test["SaleCondition"].map({"Normal":1, "Abnorml":2, "AdjLand":3, "Alloca":4, "Family":5, "Partial":6})

# for col in zeroFlagList:
#     X_test[f"{col}_flag"] = (X_test[col] == 0).astype(int)


# X_test = X_test[X_total.columns]

# #model_test_params  = {"verbosity":-1,"objective": "regression","metric": "rmse",'boosting_type': 'gbdt', 'n_estimators': 3401, 'learning_rate': 0.012403203472940544, 'num_leaves': 186, 'max_depth': 3, 'min_child_samples': 45, 'min_child_weight': 0.009180467571691761, 'min_split_gain': 0.0017618095790247808, 'subsample': 0.8861764184266921, 'colsample_bytree': 0.2741184896096117, 'reg_alpha': 0.055913323561942826, 'reg_lambda': 0.05581010414751861}

# model_test_params = {'n_estimators': 2181 , "objective": "reg:squarederror","tree_method": "hist","eval_metric": "rmse",'booster': 'gbtree', 'learning_rate': 0.011635372992720208, 'max_depth': 3, 'min_child_weight': 1.4569261205526733, 'lambda': 0.11676333549241422, 'alpha': 0.0035263692121415206, 'subsample': 0.5859003821892403, 'colsample_bytree': 0.6599059168627729, 'rate_drop': 0.4158423386888716, 'skip_drop': 0.2906315358960341}
   

# #model_test  =XGBRegressor(**model_test_params)
# #model_test = LGBMRegressor(**model_test_params)

# model_test = ElasticNet(**best_params_elastic, random_state=42)



# pipeline_test = make_pipeline(
#     preprocess_total,
#     model_test)

# pipeline_test.fit(X_total, y)

# test_preds = pipeline_test.predict(X_test)

# test_preds_reverted = np.expm1(test_preds)


# result = pd.DataFrame({"Id":test["Id"] , "SalePrice":test_preds_reverted} )
# result.to_csv("result.csv",index=False)