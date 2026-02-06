from itertools import combinations
from lightgbm import LGBMClassifier




import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

def genereate_gt_lt_num(df,num_cols,step_counts):
    num_lt_gt_combinations = {} 
    for col in num_cols:
        max = df[col].max()
        min = df[col].min()
        for step in step_counts:
            step_value = (max - min) / step
            for i in range(1,step+1):
                num_lt_gt_combinations[col+"_lt_" + str(min + step_value*i) ] = df[col] < (min + step_value*i)
                num_lt_gt_combinations[col+"_gt_" + str(max - step_value*i) ] = df[col] > (max - step_value*i)
    return num_lt_gt_combinations


def generate_category_equality(df,nominal_cols):
    nominal_eq_combinations = {}
    for col in nominal_cols:
        for value in df[col].unique():
            nominal_eq_combinations[ col + "_eq_" + str(value) ] = (df[col] == value )
    
    return nominal_eq_combinations


def generate_cross_combinations_same(d1):
    cross_comb_same = {}
    for col1, col2 in combinations(d1.keys(), 2):
        cross_comb_same[ col1 + "_and_" + col2 ] = d1[col1] & d1[col2]
        cross_comb_same[ col1 + "_or_" + col2 ] = d1[col1] | d1[col2]
    return cross_comb_same

def generate_cross_combinations_different(d1,d2):
    cross_comb_different = {}
    for col1 in d1.keys():
        for col2 in d2.keys():
            cross_comb_different[ col1 + "_and_" + col2 ] = d1[col1] & d2[col2]
            cross_comb_different[ col1 + "_or_" + col2 ] = d1[col1] | d2[col2]
    return cross_comb_different

def generate_useful_features(model,X_,y ,features):
    X = X_.copy()
    X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075,random_state=42,stratify=y)
   
    model.fit(X_train , y_train)
    #y_preds = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    baseline_score = roc_auc_score(y_valid, y_proba)
    print("baseline score:" , baseline_score)
    
    for feature in features.keys():
        X[feature] = features[feature]
        
        X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.075,random_state=42,stratify=y)
        
        model.fit(X_train , y_train)
        #y_preds = model.predict(X_valid)
        y_proba = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, y_proba)

        if score < baseline_score:
            X.pop(feature)

    return X

        
    

    
   

    



#---------------------------------------------------------------

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})


lgbm_params = {'n_estimators': 724, 'max_depth': 2, 'num_leaves': 153, 'min_child_samples': 99, 'learning_rate': 0.1387114580881059, 'subsample': 0.37549286841241186, 'colsample_bytree': 0.9077375200328026, 'reg_alpha': 0.6578963730687483, 'reg_lambda': 0.28960307157515247}

print(X.head())

numeric_cols = ["Age","BP","Cholesterol","Max HR","ST depression","Slope of ST","Number of vessels fluro"]
categorical_cols = ["Thallium","Chest pain type","EKG results"]
binary_cols = ["Sex","Exercise angina","FBS over 120"]

model = LGBMClassifier(**lgbm_params , verbose=-1)

num_features = genereate_gt_lt_num(X,numeric_cols , [6 , 5 , 5, 5, 5 , 5 , 3] )
cat_features = generate_category_equality(X,categorical_cols)

num_features_cross = generate_cross_combinations_same(num_features)
cat_features_cross = generate_cross_combinations_same(cat_features)

total = generate_cross_combinations_different(num_features,cat_features_cross).keys()

print(total)
