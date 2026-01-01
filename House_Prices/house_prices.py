import pandas as pd
import numpy as np
import seaborn as sns #type:ignore
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
from scipy.stats import randint, uniform

from xgboost import XGBRegressor


import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_rows', None)


def missing_target_analysis_numeric(df, target):
    results = []

    for col in df.columns:
        if col == target:
            continue

        if df[col].isnull().sum() == 0:
            continue

        flag = df[col].isnull().astype(int)

        grp = df.groupby(flag)[target].mean()
        mean_diff_ratio = 0.0
        # her iki grup da varsa
        if 0 in grp.index and 1 in grp.index:
            mean_not_missing = grp.loc[0]
            mean_missing = grp.loc[1]
            diff = mean_missing - mean_not_missing
        else:
            mean_not_missing = grp.get(0, None)
            mean_missing = grp.get(1, None)
            diff = None

        if mean_missing > mean_not_missing:
            mean_diff_ratio =  (mean_missing - mean_not_missing) / mean_missing
        else:
            mean_diff_ratio =  (mean_not_missing - mean_missing) / mean_not_missing
        results.append({
            'column': col,
            'dtype':df[col].dtype,
            'missing_rate': df[col].isnull().mean(),
            'target_mean_not_missing': mean_not_missing,
            'target_mean_missing': mean_missing,
            'mean_diff': diff,
            'mean_diff_ratio' : mean_diff_ratio,
        })

    return (
        pd.DataFrame(results)
        .sort_values('mean_diff', key=lambda x: x.abs(), ascending=False)
    )


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

action_list = []


##### EDA

print(train.shape)
print(train.describe())
print(train.dtypes)

print("-----------------------------")
#### gettin numerical and categorical cols

def looks_numeric_object(col, threshold=0.9):
    converted = pd.to_numeric(col, errors="coerce")
    return converted.notna().mean() >= threshold


for col in train.drop("SalePrice",axis=1).select_dtypes(include="object").columns:
    if looks_numeric_object(train[col]):
        print(f"{col} → büyük ihtimalle numerik")

num_features = []
cat_features = []

for col in train.drop("SalePrice",axis=1).columns:
    if train[col].dtype in ["int64", "float64"]:
        if train[col].nunique() < 15:
            cat_features.append(col)   # sayısal ama kategorik
        else:
            num_features.append(col)
    else:
        cat_features.append(col)

print("numeric_cols : ")
print(num_features)

print("categorical_cols")
print(cat_features)

print("-------------------------")
### MISSING VALUE ANALYSIS

# %1–5 : can be tolerated
# %30+ : serious problem
# %60+ : mostly should be deleted

# total null percentage
print(100.0 * train.isnull().sum().sum() / np.prod(train.shape) )


col_nulls = train.apply(lambda x: 100.0 * x.isnull().sum() / train.shape[0] , axis =0) # column null ratio
row_nulls = train.apply(lambda x: 100.0 * x.isnull().sum() / train.shape[1] , axis =1) # row null ratio

print(col_nulls[col_nulls > 60.0]) 
print(row_nulls[ row_nulls > 30.0]) # no need to remove any rows

#copy train
train_copy = train.copy(deep=True)
print("missing target analysis : ")
print(missing_target_analysis_numeric(train_copy,'SalePrice'))

# for col in col_nulls.index:
#     train_copy[col+'_is_missing'] = train_copy[col].isnull().astype(int)
#     print(train_copy.groupby(col+'_is_missing')['SalePrice'].mean())
#     print("---")

# action_list.append("1 - row_nulls[ row_nulls > 30.0] will be dropped")


print("-----------------------------")

# Target’ı anlamadan EDA olmaz.
# Target Analysis
target_col = train["SalePrice"]

# skewness check
# skewness > 0 sağa çarpık
# skewness < 0 sola çarpık
# skewness = 0 simetrik

skewness = target_col.skew()
print(f'Skewness: {skewness}')

# distribution 
# sns.histplot(target_col, bins=30, kde=True)
# plt.show()


# outliers

Q1 = target_col.quantile(0.25)
Q3 = target_col.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

outliers =  target_col[(target_col < lower_bound) | (target_col > upper_bound)] #df[(df['target'] < lower_bound) | (df['target'] > upper_bound)]
print(f'Outliers count: {len(outliers)}')

# sns.boxplot(x=target_col)
# plt.show()

action_list.append("2 - target is skewed , log1p will be applied on it")

# target vs features

print("numeric features vs Target:")
print(train[num_features + ["SalePrice"]].corr()['SalePrice'])

print("variance:") # low variance detect , no contribution to model
print(train[num_features].var().sort_values())

print("-----------------------------")

# print(train[num_features].describe(include='all').T)
# print(train[cat_features].describe(include='all').T)

print("-----------------------------")

# categorical feature analysis



cat_summary = pd.DataFrame({
    "missing_rate": train[cat_features].isnull().mean(),
    "n_unique": train[cat_features].nunique(),
    "n_values": train[cat_features].apply(lambda col: col.unique().tolist()),
    
}).sort_values("n_unique")

low_card = cat_summary[cat_summary.n_unique <= 10].index.tolist()

mid_card = cat_summary[
    (cat_summary.n_unique > 10) & (cat_summary.n_unique <= 30)
].index.tolist()

high_card = cat_summary[cat_summary.n_unique > 30].index.tolist()

print(cat_summary)


# numerical feature analysis
num_summary = pd.DataFrame({
    'missing_ratio': train[num_features].isnull().mean(),
    'skewness': train[num_features].skew(),
    'variance': train[num_features].var(),
    'n_unique': train[num_features].nunique()
})
print(num_summary)



################################################
# num featureları summary ye göre doldur
# 

#####################################################
num_features_test = num_features[:]
cat_features_test = cat_features[:]

X_data = train.copy(deep=True)
y_data = train["SalePrice"]
X_data.drop("SalePrice" , axis=1 , inplace=True)
X_data.drop("Id" , axis=1 , inplace=True)



num_features.remove("Id")
num_features.append("PoolArea")
cat_features.remove("PoolArea")

num_features.remove("MSSubClass")
cat_features.append("MSSubClass")


#X_data.drop(col_nulls[col_nulls > 60.0].index.to_list() , axis=1 , inplace=True)
X_data.drop("Alley",axis = 1 , inplace=True)
X_data.drop("PoolQC",axis = 1 , inplace=True)
X_data.drop("Fence",axis = 1 , inplace=True)
X_data.drop("MiscFeature",axis = 1 , inplace=True)


X_data["YearBuilt"] = X_data["YearBuilt"].max() - X_data["YearBuilt"] 
X_data["YearRemodAdd"] = X_data["YearRemodAdd"].max() - X_data["YearRemodAdd"] 
X_data.drop("GarageYrBlt",axis = 1 , inplace=True)
num_features.remove("GarageYrBlt")

###
cat_features_no_action = ['HalfBath','BsmtHalfBath','BsmtFullBath','KitchenAbvGr','FullBath','Fireplaces','GarageCars','BedroomAbvGr','OverallCond',
                        'OverallQual','MoSold','TotRmsAbvGrd','']
cat_features_nominal = ['Street','MasVnrType','BldgType','MSZoning','LotConfig','SaleCondition','GarageType','Heating','Foundation','RoofStyle',
                        'HouseStyle','RoofMatl','Condition2','Condition1','SaleType','Exterior1st','Exterior2nd','Neighborhood','YrSold','MSSubClass']

cat_features_ordinal = ['Utilities' , 'CentralAir','LandSlope','PavedDrive','GarageFinish','BsmtExposure','ExterQual','BsmtQual',
                        'LandContour','LotShape','KitchenQual','BsmtCond','ExterCond','GarageCond','GarageQual','HeatingQC',
                        'FireplaceQu','BsmtFinType1','BsmtFinType2','Functional','Electrical']
ordinal_categories = [
    ["ELO","NoSeWa","NoSewr","AllPub"],
    ['N','Y'],
    ['Gtl','Mod','Sev'],
    ['N','P','Y'],
    ['Unf','RFn','Fin'],
    ['No','Mn','Av','Gd'],
    ['Po','Fa','TA','Gd','Ex'],
    ['Low','HLS','Bnk','Lvl'],
    ['IR3','IR2','IR1','Reg'],
    ['Unf','LwQ','Rec','BLQ','ALQ','GLQ'],
    ['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],
    ['Mix','FuseP','FuseF','FuseA','SBrkr']

]
####

for col in cat_features_ordinal:
    for fList in ordinal_categories:
        for value in fList:
            X_data[col].replace(value,fList.index(value) + 1,inplace=True)
        


X_data[cat_features_ordinal] = X_data[cat_features_ordinal].fillna(value=0)
#print(X_data[cat_features_ordinal].head())

X_data = pd.get_dummies(X_data, columns=cat_features_nominal, drop_first=True)
X_data["MSSubClass_150"] = False


scaler = StandardScaler()

for col in num_features:
    X_data[col] = X_data[col].fillna(X_data[col].median())

X_data[num_features] = scaler.fit_transform(X_data[num_features])
#print(X_data.head())

model_cols = X_data.columns.tolist()

# model = LGBMRegressor(
#     n_estimators=500,
#     learning_rate=0.05,
#     max_depth=-1,
#     random_state=42
# )

# target dönüsümü
y_data = np.log1p(y_data)

X_train, X_val, y_train, y_val = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

# hyperparameter optimization
# model.fit(X_train, y_train)




### LGBM
model = LGBMRegressor(
    n_estimators=150,   # ⬅ SABİT
    random_state=42,
    verbosity=-1
)

param_dist = {
    "num_leaves": [15, 31, 63],
    "max_depth": [3, 5, -1],
    "min_child_samples": [5, 10, 20],
    "learning_rate": [0.05, 0.1],
}

rs = GridSearchCV(
    estimator=model,
    param_grid=param_dist,
    cv=2,                   # ⬅ 2 yeter
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

rs.fit(X_train, y_train)

best_model = rs.best_estimator_
#rs.best_params_


preds = best_model.predict(X_val)

print("MAE:", mean_absolute_error(y_val, preds))
print("R2 :", r2_score(y_val, preds))




def pipeline(test , num_features , cat_features , model):
    X_data = test.copy(deep=True)
    X_data.drop("Id" , axis=1 , inplace=True)

    num_features.remove("Id")
    num_features.append("PoolArea")
    cat_features.remove("PoolArea")

    num_features.remove("MSSubClass")
    cat_features.append("MSSubClass")
    #X_data.drop(col_nulls[col_nulls > 60.0].index.to_list() , axis=1 , inplace=True)
    X_data.drop("Alley",axis = 1 , inplace=True)
    X_data.drop("PoolQC",axis = 1 , inplace=True)
    X_data.drop("Fence",axis = 1 , inplace=True)
    X_data.drop("MiscFeature",axis = 1 , inplace=True)

    X_data["YearBuilt"] = X_data["YearBuilt"].max() - X_data["YearBuilt"] 
    X_data["YearRemodAdd"] = X_data["YearRemodAdd"].max() - X_data["YearRemodAdd"] 
    X_data.drop("GarageYrBlt",axis = 1 , inplace=True)
    num_features.remove("GarageYrBlt")

    ###
    cat_features_no_action = ['HalfBath','BsmtHalfBath','BsmtFullBath','KitchenAbvGr','FullBath','Fireplaces','GarageCars','BedroomAbvGr','OverallCond',
                            'OverallQual','MoSold','TotRmsAbvGrd','']
    cat_features_nominal = ['Street','MasVnrType','BldgType','MSZoning','LotConfig','SaleCondition','GarageType','Heating','Foundation','RoofStyle',
                            'HouseStyle','RoofMatl','Condition2','Condition1','SaleType','Exterior1st','Exterior2nd','Neighborhood','YrSold','MSSubClass']

    cat_features_ordinal = ['Utilities' , 'CentralAir','LandSlope','PavedDrive','GarageFinish','BsmtExposure','ExterQual','BsmtQual',
                            'LandContour','LotShape','KitchenQual','BsmtCond','ExterCond','GarageCond','GarageQual','HeatingQC',
                            'FireplaceQu','BsmtFinType1','BsmtFinType2','Functional','Electrical']
    ordinal_categories = [
        ["ELO","NoSeWa","NoSewr","AllPub"],
        ['N','Y'],
        ['Gtl','Mod','Sev'],
        ['N','P','Y'],
        ['Unf','RFn','Fin'],
        ['No','Mn','Av','Gd'],
        ['Po','Fa','TA','Gd','Ex'],
        ['Low','HLS','Bnk','Lvl'],
        ['IR3','IR2','IR1','Reg'],
        ['Unf','LwQ','Rec','BLQ','ALQ','GLQ'],
        ['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],
        ['Mix','FuseP','FuseF','FuseA','SBrkr']

    ]
    ####

    for col in cat_features_ordinal:
        for fList in ordinal_categories:
            for value in fList:
                X_data[col].replace(value,fList.index(value) + 1,inplace=True)
            


    X_data[cat_features_ordinal] = X_data[cat_features_ordinal].fillna(value=0)
    #print(X_data[cat_features_ordinal].head())

    X_data = pd.get_dummies(X_data, columns=cat_features_nominal, drop_first=True)



    scaler = StandardScaler()

    for col in num_features:
        X_data[col] = X_data[col].fillna(X_data[col].median())

    X_data[num_features] = scaler.fit_transform(X_data[num_features])

    missing = ['Heating_GasA', 'Heating_OthW', 'HouseStyle_2.5Fin', 'RoofMatl_CompShg', 'RoofMatl_Membran', 
     'RoofMatl_Metal', 'RoofMatl_Roll', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 
     'Exterior1st_ImStucc', 'Exterior1st_Stone', 'Exterior2nd_Other']
    
    for col in missing:
        X_data[col] = False


    # print( [item for item in model_cols if item not in X_data.columns.tolist()]  )

    # print( [item for item in X_data.columns.tolist() if item not in model_cols ]  )

    preds = model.predict(X_data)
    preds = np.expm1(preds)
    result  = pd.DataFrame({"Id":test["Id"].to_list() , "SalePrice":preds})
    return (result)

pipeline(test,num_features_test,cat_features_test,best_model).to_csv("result.csv") 

