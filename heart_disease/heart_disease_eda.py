import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import clone
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score, train_test_split, KFold , StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import mean_absolute_error, r2_score , accuracy_score , classification_report ,  precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder , StandardScaler
from xgboost import XGBClassifier 
from category_encoders import TargetEncoder

from itertools import combinations

from sklearn.feature_selection import mutual_info_classif


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
train_raw = train.copy()

print(train.shape)
print(test.shape)

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

X['y'] = y

X["EKG results"] = X["EKG results"].map({ 0:2 , 1:1 ,2:0 })

print(X.head())

age_bins = [-np.inf , 35.0 , 45.0 , 50.0 , 55.0 , 60.0 , 65.0 , np.inf]
X["Age_bins"] = pd.cut(X['Age'], bins=age_bins)


BP_bins = [-np.inf , 40.0 , 60.0 , 130.0 , 140.0 , 180.0 , np.inf]
X["BP_bins"] = pd.cut(X["BP"] , bins = BP_bins )

Cholesterol_bins = [-np.inf , 200.0 , 240.0 , 280.0 , 300.0 , np.inf]
X["Cholesterol_bins"] = pd.cut(X["Cholesterol"]  , bins = Cholesterol_bins )

Max_HR_bins = [-np.inf , 80.0 , 100.0 , 160.0 , 180.0 , np.inf]
X["Max_HR_bins"] = pd.cut( X["Max HR"] , bins = Max_HR_bins )




X["Age_squared"] = X["Age"] * X["Age"]
X["BP_squared"] = X["BP"] * X["BP"]
X["Cholesterol_squared"] = X["Cholesterol"] * X["Cholesterol"]
X["Max_HR_squared"] = X["Max HR"] * X["Max HR"]
X["ST_depression_squared"] = X["ST depression"] * X["ST depression"]
X["Slope_of_ST_squared"] = X["Slope of ST"] * X["Slope of ST"]
X["Number_of_vessels_fluro_squared"] = X["Number of vessels fluro"] * X["Number of vessels fluro"]

X["Age_log"] = np.log1p(X["Age"])
X["BP_log"] = np.log1p(X["BP"])
X["Cholesterol_log"] = np.log1p(X["Cholesterol"])
X["Max_HR_log"] = np.log1p(X["Max HR"])
X["ST_depression_log"] = np.log1p(X["ST depression"])
X["Slope_of_ST_log"] = np.log1p( X["Slope of ST"] )
X["Number_of_vessels_fluro_log"] = np.log1p( X["Number of vessels fluro"] )



#interactions
X["f1"] = X["EKG results"] * X["ST depression"]
X["f2"] = X["EKG results"] * X["ST depression"] * X["ST depression"]
X["f3"] = X["EKG results"] * X["ST depression"] * X["Slope of ST"]

X["f4"] = X["EKG results"] * X["ST_depression_log"] * X["Slope of ST"]
X["f5"] = X["EKG results"] * X["ST_depression_log"] * np.expm1(X["Slope of ST"])
X["f6"] = X["EKG results"] * X["ST_depression_log"] + X["Slope of ST"]

X["f7"] = X["EKG results"] * X["Number of vessels fluro"] 
X["f8"] = X["EKG results"] * X["Number of vessels fluro"] * X["Exercise angina"] 
X["f9"] = X["EKG results"] * X["Number of vessels fluro"] * X["Exercise angina"] * X["FBS over 120"]

X["f10"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Sex"] == 1)
X["f11"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f12"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f13"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)

X["f14"] =   (X["Sex"] == 1) & (X["Chest pain type"] == 3) & (X["EKG results"] == 2)

X["f15"] = (X["Thallium"] == 3) & (X["Age"] < 53.00)

#new_features = ["f"+str(i) for i in range(1,9)]



# ordinal_cols = ["EKG results","Number of vessels fluro","Exercise angina","FBS over 120"]
# nominal_cols = ["Sex","Chest pain type","Thallium"] + ["Age_bins","BP_bins","Cholesterol_bins", "Max_HR_bins"]
# #numerical_cols = ["Age","BP","Cholesterol","Max HR","ST depression","Slope of ST"]
# numerical_cols = [ x for x in X.columns.tolist() if x not in ordinal_cols+nominal_cols ]

# 1. Create feature groups

# 1.1 base features
base_features =  train.drop(["id","Heart Disease"], axis=1).columns.tolist()

#1.2 numeric transforms
numeric_transforms = [
    col for col in X.columns 
    if "_squared" in col or "_log" in col
]

#1.3 bin features
bin_features = [col for col in X.columns if "_bins" in col]

#1.4 interaction features
interaction_features = ["f1","f2","f3","f4","f5","f6","f7","f8","f9"]
my_features = ["f10","f11","f12","f13" , "f14","f15"]

#1.5 category statistics

cat_cols = ["Sex","Chest pain type" , "Thallium","Exercise angina","FBS over 120"]

for col in cat_cols:
    stats = X.groupby(col)["y"].mean()
    X[f"{col}_target_mean"] = X[col].map(stats)

category_stats = [col for col in X.columns if "_target_mean" in col]

#1.6 cross features

X["Sex_Chest"] = X["Sex"].astype(str) + "_" + X["Chest pain type"].astype(str)
freq = X["Sex_Chest"].value_counts()
X["Sex_Chest_freq"] = X["Sex_Chest"].map(freq)

#1.7 clustering features

num_cols = ["Age","BP","Cholesterol","Max HR","ST depression"]

kmeans = KMeans(n_clusters=5, random_state=42)
X["cluster"] = kmeans.fit_predict(X[num_cols])
X["cluster_target_mean"] = X.groupby("cluster")["y"].transform("mean")


#2. Group-based test

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def evaluate(features):
    model1 = LGBMClassifier(
        num_leaves=15,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
        verbosity = -1
    )
    
    model2 = LGBMClassifier(
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
        verbosity = -1
    )

    model3 = LGBMClassifier(
        num_leaves=63,
        max_depth=-1,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
        verbosity = -1
    )

    model4 = LGBMClassifier(
        num_leaves=31,
        min_child_samples=50,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
        verbosity = -1
    )
        

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores1 = cross_val_score(model1, X[features], y, cv=cv, scoring="roc_auc")
    scores2 = cross_val_score(model2, X[features], y, cv=cv, scoring="roc_auc")
    scores3 = cross_val_score(model3, X[features], y, cv=cv, scoring="roc_auc")
    scores4 = cross_val_score(model4, X[features], y, cv=cv, scoring="roc_auc")
    return [scores1.mean() , scores2.mean() , scores3.mean() , scores4.mean() ], [scores1.std() , scores2.std() , scores3.std() , scores4.std() , ]

#base
base_mean, base_std = evaluate(base_features)

#base + numeric transforms
mean1, std1 = evaluate(base_features + numeric_transforms)

#base + interactions
mean2, std2 = evaluate(base_features + interaction_features)

#base + category stats
mean3, std3 = evaluate(base_features + category_stats)

#base + clustering
mean4, std4 = evaluate(base_features + ["cluster","cluster_target_mean"])

#noise test
#Random 20 tane noise feature ekle:
noise_features = []
for i in range(20):
    X[f"noise_{i}"] = np.random.randn(len(X))
    noise_features.append(f"noise_{i}")

mean5,std5 = evaluate(base_features + noise_features)

# base + my_features
mean6,std6 = evaluate(base_features + my_features)

#decision criteria: (mean_new - mean_base) > base_std

print("base : ",base_mean , base_std)
print("base+numeric_transforms : ",mean1 ,std1)
print("base+interactions : ",mean2 , std2)
print("base+category_stats : ",mean3 , std3)
print("base+clustering: ",mean4 , std4)
print("base+noise: ",mean5 , std5)
print("base+my_features: ",mean6 , std6)



#3

# from sklearn.inspection import permutation_importance

# model = LGBMClassifier(random_state=42,verbosity=-1)
# model.fit(X[selected_features], y)

# result = permutation_importance(
#     model,
#     X[selected_features],
#     y,
#     scoring="roc_auc",
#     n_repeats=10,
#     random_state=42
# )

# importance_df = pd.DataFrame({
#     "feature": selected_features,
#     "importance": result.importances_mean
# }).sort_values("importance", ascending=False)

# print(importance_df)