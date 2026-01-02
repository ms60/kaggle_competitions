import pandas as pd
import numpy as np
import seaborn as sns #type:ignore
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from lightgbm import LGBMRegressor , LGBMClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, r2_score , accuracy_score , classification_report ,  precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
from scipy.stats import randint, uniform

from xgboost import XGBRegressor


import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_rows', None)

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print(train.head())
print(train.shape)
print(train.isnull().sum())

## FEATURE ENGINEERING

train["PassengerId_F"] = train["PassengerId"].str.split("_").str[0]
train["PassengerId_S"] = train["PassengerId"].str.split("_").str[1]


train["Cabin_F"] = train["Cabin"].str.split("/").str[0] 
train["Cabin_S"] = train["Cabin"].str.split("/").str[1] 
train["Cabin_T"] = train["Cabin"].str.split("/").str[2] 

train["Surname"] = train["Name"].str.split(" ").str[-1]

bins = [0 , 4, 8, 14, 18 , 45 , 60]
group = ['AGE_4', 'AGE_4_8', 'AGE_8_14' , 'AGE_14_18' , 'AGE_18_45' , 'AGE_60']
train['AGE_GROUP'] = pd.cut(train['Age'], bins=bins, labels=group)

#train["Total_Spend"] = train["RoomService"] + train["FoodCourt"] + train["ShoppingMall"] + train["Spa"] + train["VRDeck"]

print(train.head())
print(train.dtypes)

#####
# EDA

cat_features = [col for col in train.columns if train[col].dtype =='object' or train[col].dtype =='category' or train[col].dtype =='bool']
cat_features.remove("Transported")
cat_summary = pd.DataFrame({
    "missing_rate": train[cat_features].isnull().mean(),
    "n_unique": train[cat_features].nunique(),
    "n_values": train[cat_features].apply(lambda col: col.unique().tolist()),
    
}).sort_values("n_unique")

print(cat_summary)
#--
num_features = [col for col in train.columns if train[col].dtype == 'float64' or train[col].dtype == 'int64']
num_features.remove("Age")
print(num_features)

num_summary = pd.DataFrame({
    'missing_ratio': train[num_features].isnull().mean(),
    'skewness': train[num_features].skew(),
    'variance': train[num_features].var(),
    'n_unique': train[num_features].nunique()
})
print(num_summary)
print(train[num_features].describe().T)

#####################

print(train["Transported"].value_counts())

print(train["PassengerId_S"].value_counts())

print(train[["PassengerId_S","Transported"]].groupby("PassengerId_S").sum("Transported") )
print(train[["PassengerId_S","Transported"]].groupby("PassengerId_S").mean("Transported") )
print(train[["VIP","Transported"]].groupby("VIP").sum("Transported"))
print(train[["VIP","Transported"]].groupby("VIP").mean("Transported"))

#######################

cat_features_nominal = ["CryoSleep","VIP","Cabin_T","Destination","HomePlanet","PassengerId_S","Cabin_F" , "AGE_GROUP"]



#####################################
X_data = train[cat_features_nominal + num_features]
y_data = train["Transported"]
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

print(X_train.shape)
print(y_train.shape)


# fill missing values

#fill categorical missing
for col in num_features:
    X_train[col] = X_train[col].fillna(X_train[col].median())

for col in cat_features_nominal:
    X_train[col] = X_train[col].fillna(X_train[col].mode())

for col in num_features:
    X_test[col] = X_test[col].fillna(X_test[col].median())

for col in cat_features_nominal:
    X_test[col] = X_test[col].fillna(X_test[col].mode())


ohe_Xtrain = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

ohe_Xtest = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

X_train_cat_features_nominal_encoded = ohe_Xtrain.fit_transform(X_train[cat_features_nominal])
X_train_cat_features_nominal_colnames = [f'{col}_{cat}' for i, col in enumerate(cat_features_nominal) for cat in ohe_Xtrain.categories_[i]]
X_train_one_hot_features = pd.DataFrame(X_train_cat_features_nominal_encoded, columns=X_train_cat_features_nominal_colnames ,  index=X_train[cat_features_nominal].index)


X_test_cat_features_nominal_encoded = ohe_Xtest.fit_transform(X_test[cat_features_nominal])
X_test_cat_features_nominal_colnames = [f'{col}_{cat}' for i, col in enumerate(cat_features_nominal) for cat in ohe_Xtest.categories_[i]]
X_test_one_hot_features = pd.DataFrame(X_test_cat_features_nominal_encoded, columns=X_test_cat_features_nominal_colnames , index = X_test[cat_features_nominal].index)

X_test_one_hot_features["PassengerId_S_08"] = False
X_test_one_hot_features["Cabin_F_T"] = False

scaler_xtrain = StandardScaler()
scaler_xtest = StandardScaler()


X_train[num_features] =  scaler_xtrain.fit_transform(X_train[num_features])

X_test[num_features] =  scaler_xtest.fit_transform(X_test[num_features])


X_train_final = pd.concat([X_train_one_hot_features, X_train[num_features]], axis=1)

X_test_final = pd.concat([ X_test_one_hot_features, X_test[num_features]] , axis = 1)


print(X_train_final.shape)
print(y_train.shape)

print(len(X_train_final.columns))
print(len(X_test_final.columns))

print([col for col in X_train_cat_features_nominal_colnames if col not in X_test_cat_features_nominal_colnames])
print([col for col in X_test_cat_features_nominal_colnames if col not in X_train_cat_features_nominal_colnames])

param_dist = {
    "num_leaves": [31, 63],         # küçük ağaçlar
    "max_depth": [15, 20],           # derinlik sınırlı
    "learning_rate": [0.1],         # hızlı öğrenme
    "n_estimators": [50, 100],      # az ağaç
    "min_child_samples": [20, 50],  # küçük yaprakları engelle
    "subsample": [0.8, 1.0],        # satır örnekleme
    "colsample_bytree": [0.8, 1.0]  # feature örnekleme
}

model = LGBMClassifier(
    objective="binary",
    random_state=42,
    n_jobs=1,        # tüm CPU yerine 1 kullan → paralellik bazen RandomizedSearchCV ile sorun çıkarıyor
    verbose=-1       # console logları kapat
)

rs = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10,         # 10 kombinasyon → 1-2 dk sürer
    scoring="roc_auc",
    cv=3,              # 2-fold → hız kazanır
    verbose=0,
    n_jobs=1,          # paralellik kapalı → loop riskini azaltır
    random_state=42
)
rs.fit(X_train_final, y_train)
#model.fit(X_train_final, y_train)
best_model = rs.best_estimator_

y_pred = best_model.predict(X_test_final)


y_proba = best_model.predict_proba(X_test_final)[:, 1]

# thresholds = np.linspace(0.1, 0.9, 81)

# rows = []
# for t in thresholds:
#     y_pred_t = (y_proba >= t).astype(int)
#     rows.append({
#         "threshold": t,
#         "precision": precision_score(y_test, y_pred_t),
#         "recall": recall_score(y_test, y_pred_t),
#         "f1": f1_score(y_test, y_pred_t)
#     })


# df_thr = pd.DataFrame(rows)
# df_thr.sort_values("f1", ascending=False).head()
# print(df_thr)

print({
    "accuracy": accuracy_score(y_test, y_pred),
    "precision":precision_score(y_test, y_pred),
    "recall":recall_score(y_test, y_pred),
    "f1":f1_score(y_test, y_pred),
    "roc_auc":roc_auc_score(y_test, y_proba),
    "pr_auc":average_precision_score(y_test,y_proba)
})



###########################################################


## FEATURE ENGINEERING

test["PassengerId_F"] = test["PassengerId"].str.split("_").str[0]
test["PassengerId_S"] = test["PassengerId"].str.split("_").str[1]


test["Cabin_F"] = test["Cabin"].str.split("/").str[0] 
test["Cabin_S"] = test["Cabin"].str.split("/").str[1] 
test["Cabin_T"] = test["Cabin"].str.split("/").str[2] 

test["Surname"] = test["Name"].str.split(" ").str[-1]

bins = [0 , 4, 8, 14, 18 , 45 , 60]
group = ['AGE_4', 'AGE_4_8', 'AGE_8_14' , 'AGE_14_18' , 'AGE_18_45' , 'AGE_60']
test['AGE_GROUP'] = pd.cut(test['Age'], bins=bins, labels=group)


for col in num_features:
    test[col] = test[col].fillna(test[col].median())

for col in cat_features_nominal:
    test[col] = test[col].fillna(test[col].mode())


ohe_test = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

test_cat_features_nominal_encoded = ohe_test.fit_transform(test[cat_features_nominal])
test_cat_features_nominal_colnames = [f'{col}_{cat}' for i, col in enumerate(cat_features_nominal) for cat in ohe_test.categories_[i]]
test_one_hot_features = pd.DataFrame(test_cat_features_nominal_encoded, columns=test_cat_features_nominal_colnames ,  index=test[cat_features_nominal].index)


scaler_test = StandardScaler()

test[num_features] =  scaler_test.fit_transform(test[num_features])

test_final = pd.concat([ test_one_hot_features, test[num_features]] , axis = 1)

test_pred = best_model.predict(test_final)

result  = pd.DataFrame({"PassengerId":test["PassengerId"].to_list() , "Transported":test_pred})
result.to_csv("result.csv")