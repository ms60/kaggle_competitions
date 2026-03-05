# greedy_boolean_builder.py

import os
import gc
import random
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from itertools import combinations, product
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from tqdm import tqdm

import resource

# Linux / WSL only
resource.setrlimit(resource.RLIMIT_AS, (12_000_000_000, 12_000_000_000))

gc.collect()

def hard_cleanup(*objs):
    for o in objs:
        try:
            del o
        except:
            pass
    gc.collect()


skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=60)

# =====================================================
# FEATURE GENERATORS (WITH EXPRESSIONS)
# =====================================================

def generate_gt_lt_num(df, num_cols, step_counts):
    feats = {}
    for col, step in zip(num_cols, step_counts):
        mn, mx = df[col].min(), df[col].max()
        step_val = (mx - mn) / step

        for i in range(1, step):
            lt = mn + step_val * i
            gt = mx - step_val * i

            feats[f"{col}_lt_{lt:.2f}"] = (
                df[col] < lt,
                f'(X["{col}"] < {lt:.2f})'
            )
            feats[f"{col}_gt_{gt:.2f}"] = (
                df[col] > gt,
                f'(X["{col}"] > {gt:.2f})'
            )
    return feats


def generate_category_equality(df, cat_cols):
    feats = {}
    for col in cat_cols:
        for v in df[col].dropna().unique():
            feats[f"{col}_eq_{v}"] = (
                df[col] == v,
                f'(X["{col}"] == {repr(v)})'
            )
    return feats


def generate_cross_combinations_same(feats, ops=("and",)):
    out = {}
    keys = list(feats.keys())
    for k1, k2 in combinations(keys, 2):
        v1, e1 = feats[k1]
        v2, e2 = feats[k2]

        if "and" in ops:
            out[f"{k1}_AND_{k2}"] = (
                v1 & v2,
                f"{e1} & {e2}"
            )
        if "or" in ops:
            out[f"{k1}_OR_{k2}"] = (
                v1 | v2,
                f"{e1} | {e2}"
            )
    return out


def generate_cross_combinations_different(f1, f2, ops=("and",)):
    out = {}
    for (k1, (v1, e1)), (k2, (v2, e2)) in product(f1.items(), f2.items()):
        if "and" in ops:
            out[f"{k1}_AND_{k2}"] = (
                v1 & v2,
                f"{e1} & {e2}"
            )
        if "or" in ops:
            out[f"{k1}_OR_{k2}"] = (
                v1 | v2,
                f"{e1} | {e2}"
            )
    return out


# =====================================================
# GREEDY AUC SELECTION
# =====================================================

def greedy_auc_feature_addition(
    X,
    y,
    candidate_features,
    model,
    test_size=0.075,
    random_state=42
):
    X_work = X.copy()

    # X_tr, X_val, y_tr, y_val = train_test_split(
    #     X_work,
    #     y,
    #     test_size=test_size,
    #     stratify=y,
    #     random_state=random_state
    # )

    # model.fit(X_tr, y_tr)

    # base_auc = roc_auc_score(
    #     y_val,
    #     model.predict_proba(X_val)[:, 1]
    # )
    
    base_auc = cross_val_score(
        model,
        X_work, y,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1
    ).min()

    print(f"Initial AUC: {base_auc:.5f}")

    accepted = {}

    for name, (col, expr) in tqdm(
        candidate_features.items(),
        total=len(candidate_features),
        desc="Evaluating feature combinations"
    ):
        X_work[name] = col.astype("int8")

        # X_tr, X_val, y_tr, y_val = train_test_split(
        #     X_work,
        #     y,
        #     test_size=test_size,
        #     stratify=y,
        #     random_state=random_state
        # )

        # model.fit(X_tr, y_tr)

        # auc = roc_auc_score(
        #     y_val,
        #     model.predict_proba(X_val)[:, 1]
        # )

        auc = cross_val_score(
            model,
            X_work, y,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1
        ).min()

        if auc > base_auc:
            base_auc = auc
            accepted[name] = (col, expr)
            tqdm.write(f"[KEEP] X['f1'] = {expr} → AUC {auc:.5f}")
        else:
            X_work.drop(columns=[name], inplace=True)

    print(f"\nSelected {len(accepted)} features")
    print(f"Final AUC: {base_auc:.5f}")

    return accepted, X_work


# =====================================================
# RUN
# =====================================================

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id", axis=1)
y = X.pop("Heart Disease")
y = y.map({"Presence": 1, "Absence": 0})

X['f1'] = (X["Age"] < 32.00) & (X["Thallium"] == np.int64(7)) & (X["Exercise angina"] == 1) #→ AUC 0.95459                                                                           
X['f2'] = (X["Age"] > 68.00) & (X["Thallium"] == np.int64(3)) & (X["Exercise angina"] == 1) #→ AUC 0.95460                                                                           
X['f3'] = (X["Age"] > 68.00) & (X["Chest pain type"] == np.int64(2)) & (X["Sex"] == 1) #→ AUC 0.95461                                                                                
X['f4'] = (X["Age"] < 53.00) & (X["EKG results"] == np.int64(0)) & (X["Sex"] == 1) #→ AUC 0.95461                                                                                    
X['f5'] = (X["Age"] < 56.00) & (X["EKG results"] == np.int64(2)) & (X["Sex"] == 1) #→ AUC 0.95462                                                                                    
X['f6'] = (X["Age"] < 59.00) & (X["Thallium"] == np.int64(7)) & (X["Sex"] == 1) #→ AUC 0.95462                                                                                       
X['f7'] = (X["Age"] > 47.00) & (X["Thallium"] == np.int64(3)) & (X["Sex"] == 1) #→ AUC 0.95464  

X['f8'] = (X["Age"] < 32.00) & (X["Thallium"] == np.int64(7)) & (X["Exercise angina"] == 1) #→ AUC 0.95459     
X['f9'] = (X["Age"] < 32.00) & (X["Thallium"] == np.int64(7)) & (X["Exercise angina"] == 1) #→ AUC 0.95459  
X['f10'] = (X["Age"] < 32.00) & (X["Thallium"] == np.int64(7)) & (X["Exercise angina"] == 1) #→ AUC 0.95459

X['f11'] = (X["BP"] < 104.60) & (X["Slope of ST"] == np.int64(2))# → AUC 0.94979                                                                                                      
X['f12'] = (X["BP"] < 104.60) & (X["Slope of ST"] == np.int64(1))# → AUC 0.95332 

                                                                                                    
X['f13'] = (X["BP"] < 104.60) & (X["Slope of ST"] == np.int64(1))# → AUC 0.95332 

X['f14'] = (X["Slope of ST"] == np.int64(2)) & (X["Thallium"] == np.int64(7))# → AUC 0.95303                                                                                          
X['f15'] = (X["Slope of ST"] == np.int64(2)) & (X["Thallium"] == np.int64(3))# → AUC 0.95310   

X['f16'] = (X["Slope of ST"] == np.int64(2)) & (X["Thallium"] == np.int64(7))

X['f17'] = (X["Number of vessels fluro"] == np.int64(2)) & (X["EKG results"] == np.int64(0))

X['f18'] = (X["Exercise angina"] == 1) & (X["Number of vessels fluro"] == np.int64(2)) & (X["EKG results"] == np.int64(0))# → AUC 0.95500 
X['f19'] = (X["Exercise angina"] == 1) & (X["Number of vessels fluro"] == np.int64(2)) & (X["EKG results"] == np.int64(2))# → AUC 0.95505

X['f20'] = (X["Exercise angina"] == 1) & (X["Number of vessels fluro"] == np.int64(2)) & (X["Thallium"] == np.int64(3))# → AUC 0.95505                                                
X['f21'] = (X["Exercise angina"] == 1) & (X["Number of vessels fluro"] == np.int64(2)) & (X["Thallium"] == np.int64(6))# → AUC 0.95506 
X['f22'] = (X["Exercise angina"] == 1) & (X["Number of vessels fluro"] == np.int64(0)) & (X["Chest pain type"] == np.int64(3))# → AUC 0.95507

X['f23'] = (X["Sex"] == 1) & (X["Number of vessels fluro"] == np.int64(2)) & (X["Chest pain type"] == np.int64(4))# → AUC 0.95506

X['f24'] = (X["Sex"] == 1) & (X["Number of vessels fluro"] == np.int64(2)) & (X["EKG results"] == np.int64(0))# → AUC 0.95502

X['f25'] = (X["Sex"] == 1) & (X["Number of vessels fluro"] == np.int64(2)) & (X["Chest pain type"] == np.int64(3))# → AUC 0.95473

# X["f1"] = (X["Age"] < 32.00) & (X["Thallium"] == np.int64(3)) & (X["Sex"] == 1)
# X["f2"] = (X["Age"] > 74.00) & (X["Thallium"] == np.int64(3)) & (X["Sex"] == 1)
# X["f3"] = (X["Age"] < 35.00) & (X["Thallium"] == np.int64(7)) & (X["Exercise angina"] == 1)
# X["f4"] = (X["Age"] < 35.00) & (X["Thallium"] == np.int64(7)) & (X["Sex"] == 1)
# X["f5"] = (X["Age"] > 56.00) & (X["Thallium"] == np.int64(6)) & (X["Sex"] == 1)

# X["f1"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Sex"] == 1)
# X["f2"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
# X["f3"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
# X["f4"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)

# X["f5"] =   (X["Sex"] == 1) & (X["Chest pain type"] == 3) & (X["EKG results"] == 2)

# X["f6"] = (X["Thallium"] == 3) & (X["Age"] < 53.00)


#X['f7'] = (X["Number of vessels fluro"] == np.int64(3)) & (X["Max HR"] > 103.75) & (X["Sex"] == 1) #→ AUC 0.95656                                                                       
#X['f7'] = (X["Number of vessels fluro"] == np.int64(0)) & (X["ST depression"] > 1.86) & (X["Sex"] == 1)# → AUC 0.95657                                                                  
#X['f7'] = (X["Slope of ST"] == np.int64(3)) & (X["ST depression"] < 0.62) & (X["Sex"] == 1) #→ AUC 0.95658                                                                              
#X['f7'] = (X["Slope of ST"] == np.int64(3)) & (X["Cholesterol"] > 235.50) & (X["Exercise angina"] == 1)# → AUC 0.95660                                                                  
#X['f7'] = (X["Number of vessels fluro"] == np.int64(0)) & (X["BP"] < 120.50) & (X["Sex"] == 1)# → AUC 0.95660                                                                           
#X['f7'] = (X["Slope of ST"] == np.int64(2)) & (X["ST depression"] > 1.24) & (X["Sex"] == 1)# → AUC 0.95660                                                                              
#X['f7'] = (X["Slope of ST"] == np.int64(3)) & (X["Age"] < 61.00) & (X["Sex"] == 1) #→ AUC 0.95663                                                                                       
#X['f7'] = (X["Chest pain type"] == np.int64(2)) & (X["Age"] > 45.00) & (X["Sex"] == 1) #→ AUC 0.95663                                                                                   
#X['f7'] = (X["Chest pain type"] == np.int64(1)) & (X["Age"] < 45.00) & (X["Exercise angina"] == 1) #→ AUC 0.95665  

# for i in range(1,7):
#     X[f"f{i}"] = X[f"f{i}"].astype(int)

# X["f7"] = (X["Age"] > 68.00) & (X["Thallium"] == np.int64(7)) & (X["FBS over 120"] == 1)
# X["f8"] = (X["Age"] > 65.00) & (X["Thallium"] == np.int64(7)) & (X["FBS over 120"] == 1)

# [KEEP] (X["Age"] > 68.00) & (X["Thallium"] == np.int64(7)) & (X["FBS over 120"] == 1) → AUC 0.95673                                                                                           
# [KEEP] (X["Age"] > 65.00) & (X["Thallium"] == np.int64(7)) & (X["FBS over 120"] == 1) → AUC 0.95674   

# [KEEP] X['f1'] = (X["ST depression_disc"] == 'f_0_bin_4') & (X["EKG results"] == np.int64(0)) → AUC 0.95482                                                                                   
# [KEEP] X['f1'] = (X["ST depression_disc"] == 'f_0_bin_4') & (X["Number of vessels fluro"] == np.int64(1)) → AUC 0.95483                                                                       
# [KEEP] X['f1'] = (X["ST depression_disc"] == 'f_0_bin_2') & (X["Max HR_disc"] == 'f_3_bin_3') → AUC 0.95483                                                                                   
# [KEEP] X['f1'] = (X["ST depression_disc"] == 'f_0_bin_3') & (X["Age_disc"] == 'f_1_bin_4') → AUC 0.95483                                                                                      
# [KEEP] X['f1'] = (X["ST depression_disc"] == 'f_0_bin_3') & (X["Cholesterol_disc"] == 'f_2_bin_4') → AUC 0.95484 
# 
# 

# [KEEP] X['f1'] = (X["BP_disc"] == 'f_4_bin_2') & (X["Slope of ST"] == np.int64(2)) → AUC 0.95482                                                                                              
# [KEEP] X['f1'] = (X["ST depression_disc"] == 'f_0_bin_3') & (X["Slope of ST"] == np.int64(2)) → AUC 0.95482                                                                                   
# [KEEP] X['f1'] = (X["Max HR_disc"] == 'f_3_bin_2') & (X["Number of vessels fluro"] == np.int64(0)) → AUC 0.95483                                                                              
# [KEEP] X['f1'] = (X["Age_disc"] == 'f_1_bin_1') & (X["Number of vessels fluro"] == np.int64(1)) → AUC 0.95483                                                                                 
# [KEEP] X['f1'] = (X["BP_disc"] == 'f_4_bin_2') & (X["Thallium"] == np.int64(6)) → AUC 0.95483                                                                                                 
# [KEEP] X['f1'] = (X["Age_disc"] == 'f_1_bin_1') & (X["Chest pain type"] == np.int64(3)) → AUC 0.95483                                                                                         
# [KEEP] X['f1'] = (X["BP_disc"] == 'f_4_bin_0') & (X["Chest pain type"] == np.int64(2)) → AUC 0.95484                                                                                          
# [KEEP] X['f1'] = (X["Thallium"] == np.int64(3)) & (X["Slope of ST"] == np.int64(2)) → AUC 0.95484    



# X["f7"] =  (X["Exercise angina"] == 1) & (X["Sex"] == 1)
# X["f8"] = (X["Exercise angina"] == 1) & (X["FBS over 120"] == 1)
# X["f9"] = (X["Sex"] == 1) & (X["FBS over 120"] == 1)

# X['HR_Reserve'] = (220 - X['Age']) - X['Max HR']
# X['HR_Achievement_Ratio'] = X['Max HR'] / (220 - X['Age'])

#X.pop("BP")

#X["f7"] = (X["Number of vessels fluro"] > 0.75) & (X["Slope of ST"] > 1.50)
#[KEEP] (X["Number of vessels fluro"] > 0.75) & (X["Slope of ST"] > 1.50) → AUC 0.95664 


numeric_cols = [
    "ST depression",
    "Age",  "Cholesterol",
    "Max HR", "BP"
    
]

categorical_cols = [
    "EKG results","Thallium", "Chest pain type","Slope of ST","Number of vessels fluro"
]

binary_cols = [
    "Sex", "FBS over 120" , "Exercise angina" 
]

#-------------------------------

from sklearn.tree import DecisionTreeClassifier
def get_bins(X,y,col):
    tree = DecisionTreeClassifier(max_leaf_nodes=5, min_samples_leaf=50) 
    tree.fit(X[[col]], y) 
    thresholds = tree.tree_.threshold 

    splits = sorted(thresholds[thresholds != -2])
    bins = [-np.inf] + splits + [np.inf]
    print(col , " : " ,bins)
    return bins

X_disc = pd.DataFrame()

for ix,col in enumerate(numeric_cols):
    bins = get_bins(X,y,col)
    labels = [f"f_{ix}_bin_{i}" for i in range(len(bins)-1)]
    X_disc[ col + "_disc"] = pd.cut(X[col], bins=bins, labels=labels)

#print(X_disc.head())

X_total = pd.concat( [X_disc , X[categorical_cols]] , axis=1 )

X_total = pd.concat( [X_total , X[binary_cols]] , axis=1 )

# X_total['f1'] = (X_total["ST depression_disc"] == 'f_0_bin_4') & (X_total["EKG results"] == np.int64(0))
# X_total['f2'] = (X_total["ST depression_disc"] == 'f_0_bin_4') & (X_total["Number of vessels fluro"] == np.int64(1))
# X_total['f3'] = (X_total["ST depression_disc"] == 'f_0_bin_2') & (X_total["Max HR_disc"] == 'f_3_bin_3')
# X_total['f4'] = (X_total["ST depression_disc"] == 'f_0_bin_3') & (X_total["Age_disc"] == 'f_1_bin_4')
# X_total['f5'] = (X_total["ST depression_disc"] == 'f_0_bin_3') & (X_total["Cholesterol_disc"] == 'f_2_bin_4')

#--------------------

# [KEEP] X['f1'] = (X["Age_disc"] == 'f_1_bin_1') & (X["Number of vessels fluro"] == np.int64(1)) → AUC 0.95482                                                                                 
# [KEEP] X['f1'] = (X["Max HR_disc"] == 'f_3_bin_0') & (X["FBS over 120"] == np.int64(0)) → AUC 0.95483                                                                                         
# [KEEP] X['f1'] = (X["Max HR_disc"] == 'f_3_bin_2') & (X["Thallium"] == np.int64(6)) → AUC 0.95483                                                                                             
# [KEEP] X['f1'] = (X["BP_disc"] == 'f_4_bin_1') & (X["EKG results"] == np.int64(0)) → AUC 0.95483                                                                                              
# [KEEP] X['f1'] = (X["Chest pain type"] == np.int64(3)) & (X["Exercise angina"] == np.int64(0)) → AUC 0.95483                                                                                  
# [KEEP] X['f1'] = (X["Cholesterol_disc"] == 'f_2_bin_0') & (X["Chest pain type"] == np.int64(1)) → AUC 0.95485                                                                                 
# [KEEP] X['f1'] = (X["ST depression_disc"] == 'f_0_bin_4') & (X["Max HR_disc"] == 'f_3_bin_3') → AUC 0.95485                                                                                   
# [KEEP] X['f1'] = (X["ST depression_disc"] == 'f_0_bin_1') & (X["Exercise angina"] == np.int64(0)) → AUC 0.95485                                                                               
# [KEEP] X['f1'] = (X["Age_disc"] == 'f_1_bin_0') & (X["Slope of ST"] == np.int64(1)) → AUC 0.95485                                                                                             
# [KEEP] X['f1'] = (X["Age_disc"] == 'f_1_bin_1') & (X["BP_disc"] == 'f_4_bin_2') → AUC 0.95485                                                                                                 
# [KEEP] X['f1'] = (X["BP_disc"] == 'f_4_bin_0') & (X["Chest pain type"] == np.int64(2)) → AUC 0.95486                                                                                          
# [KEEP] X['f1'] = (X["Max HR_disc"] == 'f_3_bin_3') & (X["EKG results"] == np.int64(0)) → AUC 0.95486                                                                                          
# [KEEP] X['f1'] = (X["ST depression_disc"] == 'f_0_bin_1') & (X["Age_disc"] == 'f_1_bin_4') → AUC 0.95487                                                                                      
# [KEEP] X['f1'] = (X["ST depression_disc"] == 'f_0_bin_4') & (X["Age_disc"] == 'f_1_bin_3') → AUC 0.95487                                                                                      
# [KEEP] X['f1'] = (X["Number of vessels fluro"] == np.int64(2)) & (X["Exercise angina"] == np.int64(0)) → AUC 0.95487                                                                          
# [KEEP] X['f1'] = (X["BP_disc"] == 'f_4_bin_4') & (X["Thallium"] == np.int64(3)) → AUC 0.95488                                                                                                 
# [KEEP] X['f1'] = (X["EKG results"] == np.int64(1)) & (X["Sex"] == np.int64(0)) → AUC 0.95488                                                                                                  
# [KEEP] X['f1'] = (X["Exercise angina"] == np.int64(0)) & (X["Sex"] == np.int64(0)) → AUC 0.95489                                                                                              
# [KEEP] X['f1'] = (X["Age_disc"] == 'f_1_bin_1') & (X["FBS over 120"] == np.int64(1)) → AUC 0.95489                                                                                            
# [KEEP] X['f1'] = (X["Max HR_disc"] == 'f_3_bin_2') & (X["Slope of ST"] == np.int64(2)) → AUC 0.95490      

#print(X_total.head())

#-------------------------------




#---------------------------------


#lgbm_params = {'boosting_type': 'gbdt', 'n_estimators': 6405, 'learning_rate': 0.030752124591604243, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.620293411878579, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}
#lgbm_params = {'boosting_type': 'gbdt', 'learning_rate': 0.10139689417192685, 'num_leaves': 105, 'min_child_samples': 150, 'min_child_weight': 1.4859133739671821, 'min_split_gain': 0.015630473429093072, 'subsample': 0.22242548213805655, 'colsample_bytree': 0.12177938947172448, 'reg_alpha': 1.341517183219574, 'reg_lambda': 0.008658382822838841}
#lgbm_params = {'boosting_type': 'gbdt', 'n_estimators': 6405, 'learning_rate': 0.030752124591604243, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.620293411878579, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}
#lgbm_params = {'boosting_type': 'gbdt', 'n_estimators': 7500, 'learning_rate': 0.030752124591604243, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.620293411878579, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}
#{'boosting_type': 'gbdt', 'n_estimators': 1236, 'learning_rate': 0.15163077379677498, 'num_leaves': 97, 'max_depth': 3, 'min_child_samples': 23, 'subsample': 0.8203405205601193, 'colsample_bytree': 0.15103910849091867, 'reg_alpha': 1.8094159069409557, 'reg_lambda': 0.8206319983321384}


lgbm_params ={'n_estimators': 724, 'max_depth': 2, 'num_leaves': 153, 'min_child_samples': 99, 'learning_rate': 0.1387114580881059, 'subsample': 0.37549286841241186, 'colsample_bytree': 0.9077375200328026, 'reg_alpha': 0.6578963730687483, 'reg_lambda': 0.28960307157515247}
#lgbm_params = {'boosting_type': 'gbdt', 'n_estimators': 6405, 'learning_rate': 0.030752124591604243, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.620293411878579, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}
#Initial AUC: 0.95545 with cv
# 0.95657
#lgbm_params = {'boosting_type': 'gbdt', 'n_estimators': 8037, 'learning_rate': 0.01961744449399505, 'num_leaves': 254, 'max_depth': 5, 'min_child_samples': 14, 'min_child_weight': 0.6303846127454082, 'min_split_gain': 0.18764237497463476, 'subsample': 0.26167571300890513, 'colsample_bytree': 0.17491054511557924, 'reg_alpha': 0.007670436055874847, 'reg_lambda': 0.9952489654312465}

#lgbm_params = {'boosting_type': 'gbdt', 'n_estimators': 7730, 'learning_rate': 0.028778047251309946, 'num_leaves': 96, 'max_depth': 3, 'min_child_samples': 13, 'min_child_weight': 0.2316553378161039, 'min_split_gain': 0.042823710761769954, 'subsample': 0.36699422027167905, 'colsample_bytree': 0.12199531962565616, 'reg_alpha': 0.5624471773258042, 'reg_lambda': 2.9747133675697754}
lgbm_params.update({
    'boosting_type': 'gbdt',
    'n_estimators': 1992,
    # 'min_child_weight': 0.0023487410631235703,
    # 'min_split_gain': 0.004094962237309152,
    'min_child_weight': 0.2045358944188131, 'min_split_gain': 0.0008287110542991236,
    "random_state":110,
    "objective": "binary",
    "metric": "auc",
})


print(X.head())

# lgbm_params = {
#     "objective": "binary",
#     "metric": "auc",
#     'n_estimators': 724,
#     'max_depth': 2,
#     'num_leaves': 153,
#     'min_child_samples': 99,
#     'learning_rate': 0.1387114580881059,
#     'subsample': 0.37549286841241186,
#     'colsample_bytree': 0.9077375200328026,
#     'reg_alpha': 0.6578963730687483,
#     'reg_lambda': 0.28960307157515247
# }

model = LGBMClassifier(**lgbm_params, verbose=-1)

# model = LogisticRegression(
#     penalty="elasticnet",
#     C=1.0,
#     l1_ratio=0.5,       # 0 → Ridge, 1 → Lasso
#     solver="saga",      # şart
#     max_iter=1000,
#     n_jobs=-1
# )

# cat_params =  {
#     "loss_function": "Logloss",
#     "eval_metric": "AUC",
#     "iterations": 5000,
#     "learning_rate": 0.03,
#     "depth": 5,
#     "l2_leaf_reg": 5,

#     "boosting_type": "Plain",
#     "one_hot_max_size": 10,
#     "max_ctr_complexity": 2,

#     "subsample": 0.8,
#     "rsm": 0.8,

#     "random_seed": 42,
#     "thread_count": -1,
#     "verbose": 0
# }

# model = CatBoostClassifier(**cat_params)

    # "Number of vessels fluro",
    # "ST depression",
    # "Slope of ST", 
    # "Age", "BP", "Cholesterol",
    # "Max HR", 

selected_num_features = generate_gt_lt_num(
    X, ["Age" ,"ST depression","Number of vessels fluro" ], [16 ,31 , 3]
    #X , ["Cholesterol","BP","Max HR",] , [10 , 10 , 10]
)

categorical_cols = [
    "EKG results","Thallium", "Chest pain type","Slope of ST","Number of vessels fluro"
]



selected_cat_features = generate_category_equality(
    #X , [ "Thallium" , "EKG results" , "Chest pain type"]
    X , [ "Number of vessels fluro" , "Slope of ST"]
    #X , ["Number of vessels fluro" , "Slope of ST" ]
    )

selected_cat_features_2 = generate_category_equality(
    #X , [ "Thallium" , "EKG results" , "Chest pain type"]
    #X , [ "Slope of ST","Number of vessels fluro"]
    X , [ "EKG results" , "Chest pain type" , "Thallium"  ]
    )

    # "ST depression",
    # "Age", "BP", "Cholesterol",
    # "Max HR", 

# num_features = generate_gt_lt_num(
#     X, numeric_cols, [10, 6, 4, 4, 4]
# )

# cat_features = generate_category_equality(
#     X, categorical_cols
# )

base_booleans = {
    col: (
        X[col].astype(bool),
        f'(X["{col}"] == 1)'
    )
    for col in binary_cols
}


#selected_num_bool = generate_cross_combinations_different(selected_num_features,base_booleans)

#cat_bool = generate_cross_combinations_different(cat_features,base_booleans)

#bool_bool = generate_cross_combinations_same(base_booleans)

selected_num_num = generate_cross_combinations_same(selected_num_features)



selected_num_cat = generate_cross_combinations_different(selected_num_features,selected_cat_features)

selected_num_cat_bool = generate_cross_combinations_different(selected_num_cat , base_booleans)

selected_cat_cat = generate_cross_combinations_different(selected_cat_features,selected_cat_features_2)

selected_cat_cat_bool = generate_cross_combinations_different(base_booleans , selected_cat_cat  )



# num_cat = generate_cross_combinations_different(cat_features,num_features)
# num_cat_bool = generate_cross_combinations_different(num_cat,base_booleans)

#cat_cat_features = generate_cross_combinations_same(cat_features)


# items = list(num_cat_bool.items())
# random.shuffle(items)

# num_cat_bool_shuffled = dict(items)

# catcat_cat = generate_cross_combinations_different(cat_cat_features,cat_features)


#num_bool_features = generate_cross_combinations_different(num_features,base_booleans)

#num_num_features = generate_cross_combinations_same(num_features)
# numnum_num = generate_cross_combinations_different(num_num_features,num_features)



# total = generate_cross_combinations_different(
#     num_features,
#     cat_features
# )

# grand_total = generate_cross_combinations_different(
#     total,
#     base_booleans,
#     ops=("and",)
# )

# bool_bool = generate_cross_combinations_same(base_booleans)
# bool_bool_bool = generate_cross_combinations_different(base_booleans ,bool_bool )

# bool_bool_bool_cat = generate_cross_combinations_different(cat_features , bool_bool_bool)

# cat_cat_num = generate_cross_combinations_different(cat_cat_features , num_features)
# cat_cat_num_bool = generate_cross_combinations_different(cat_cat_num , base_booleans)

#X.pop("Max HR")

X_small, _, y_small, _ = train_test_split(
    X,
    y,
    train_size=120_000,
    stratify=y,
    shuffle=True,
    random_state=42
)


accepted, X_final = greedy_auc_feature_addition(
    X_small,
    y_small,
    selected_num_cat_bool,
    model
)


# =====================================================
# OPTIONAL: DUMP ACCEPTED RULES
# =====================================================

print("\nAccepted rules (copy-paste ready):\n")
for _, (_, expr) in accepted.items():
    print(expr)

hard_cleanup(
    #num_features,
    selected_num_cat_bool,
    #num_cat_bool_shuffled,

    X_final
)
