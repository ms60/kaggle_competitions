import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import clone
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


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

# print(X.head())
# print(y.head())


X["ind_1"] = (X["Age"] > 40 ) & ( X["Exercise angina"] )
X["ind_2"] = (X["Age"] > 50) &  (X["FBS over 120"] )
X["ind_3"] = ( X["Exercise angina"] ) &  (X["ST depression"] > 1 )
X["ind_4"] = ( X["Exercise angina"] ) & (X["FBS over 120"] )

X["ind_5"] = (X["ind_4"] ) & ( X["EKG results"] > 0 )

X["ind_6"] = X["ind_3"] &  X["ind_4"] & (X["Thallium"] == 3)

X["ind_7"] = ( X["EKG results"] > 0 ) & ( X["Chest pain type"]==3 )
X["ind_8"] = ( X["EKG results"] > 0 ) & ( X["Chest pain type"]==2 )

X["ind_9"] =  ( X["Number of vessels fluro"] > 0 ) & (X["Age"] < 40) 

X["ind_10"] =  (X["Sex"] ==1 )  & (X["Age"] > 40 ) & ( X["Cholesterol"] > 300 )
X["ind_11"] =  (X["Sex"] ==1 )  & (X["Age"] > 40 ) & ( X["BP"] > 180 )


for i in range(1,12):
    X["ind_"+str(i)] = X["ind_"+str(i)].astype(int) 



# Age > 55 AND Exercise_angina
# Max_HR < 120 AND ST_depression > 2


print(X.head())





nominal_cols = ["Sex","Chest pain type","Thallium"] 
num_cols = ["Age","BP","Cholesterol","Max HR","ST depression","Slope of ST",]
#te_cols = ["EKG results","Sex","Chest pain type","Thallium"]
ordinal_cols = ["EKG results","Number of vessels fluro"]
yes_or_no = ["FBS over 120","Exercise angina","Age_flag","BP_flag","Max_HR_flag","Cholesterol_flag"]

# for col in num_cols:
#     X[ col + "_squared" ] = X[col] * X[col]
#     X[ col + "_log"] = np.log1p( X[col] )
#     X[ col + "_sqrt" ] = np.sqrt( X[col] )

# for col in ordinal_cols:
#     X[ col + "_squared" ] = X[col] * X[col]
#     X[ col + "_log"] = np.log1p( X[col] )
#     X[ col + "_sqrt" ] = np.sqrt( X[col] )

# for col1, col2 in combinations(num_cols, 2):
#     X[col1 + "_multiply_" + col2] = X[col1] * X[col2]
#     X[col1 + "_divide_" + col2] = X[col1] / X[col2]

# for col1, col2 in combinations(ordinal_cols, 2):
#     X[col1 + "_multiply_" + col2] = X[col1] * X[col2]
#     X[col1 + "_divide_" + col2] = X[col1] / X[col2]

# for col1 in num_cols:
#     for col2 in ordinal_cols:
#         X[col1 + "_multiply_" + col2 ] = X[col1] * X[col2]
#         X[col1 + "_divide_" + col2 ] = X[col1] / X[col2]

# print(X.head())

X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075, random_state=42,stratify=y)


model = LGBMClassifier(verbosity=-1)

model.fit(X_train,y_train)
y_proba = model.predict_proba(X_valid)[:, 1]
score = roc_auc_score(y_valid, y_proba)

print(score)





