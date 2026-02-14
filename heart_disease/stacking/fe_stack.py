from itertools import combinations
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
X = train.drop("id", axis=1)
X_test = test.drop("id", axis=1)
y = X.pop("Heart Disease").map({"Presence": 1, "Absence": 0})

num_cols = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
bin_cols = ["Sex", "FBS over 120", "Exercise angina"]
nominal_cols = ["Chest pain type", "Slope of ST","Thallium"]
ordinal_cols = ["EKG results","Number of vessels fluro"]
cat_cols = nominal_cols + ordinal_cols


X_FE = pd.DataFrame()
X_FE_2 = pd.DataFrame()
X_FE_3 = pd.DataFrame()

test_FE = pd.DataFrame()
test_FE_2 = pd.DataFrame()
test_FE_3 = pd.DataFrame()

for col in num_cols:
    X_FE[ col + "_squared" ] = X[col] * X[col]
    X_FE[ col + "_log"] = np.log1p( X[col] )
    X_FE[ col + "_sqrt" ] = np.sqrt( X[col] )

for col in ordinal_cols:
    X_FE[ col + "_squared" ] = X[col] * X[col]
    X_FE[ col + "_log"] = np.log1p( X[col] )
    X_FE[ col + "_sqrt" ] = np.sqrt( X[col] )

for col1, col2 in combinations(num_cols, 2):
    X_FE[col1 + "_multiply_" + col2] = X[col1] * X[col2]
    X_FE[col1 + "_divide_" + col2] = X[col1] / X[col2]

for col1, col2 in combinations(ordinal_cols, 2):
    X_FE[col1 + "_multiply_" + col2] = X[col1] * X[col2]
    X_FE[col1 + "_divide_" + col2] = X[col1] / X[col2]

for col1 in num_cols:
    for col2 in ordinal_cols:
        X_FE[col1 + "_multiply_" + col2 ] = X[col1] * X[col2]
        X_FE[col1 + "_divide_" + col2 ] = X[col1] / X[col2]

#----

for col in num_cols:
    test_FE[ col + "_squared" ] = X_test[col] * X_test[col]
    test_FE[ col + "_log"] = np.log1p( X_test[col] )
    test_FE[ col + "_sqrt" ] = np.sqrt( X_test[col] )

for col in ordinal_cols:
    test_FE[ col + "_squared" ] = X_test[col] * X_test[col]
    test_FE[ col + "_log"] = np.log1p( X_test[col] )
    test_FE[ col + "_sqrt" ] = np.sqrt( X_test[col] )

for col1, col2 in combinations(num_cols, 2):
    test_FE[col1 + "_multiply_" + col2] = X_test[col1] * X_test[col2]
    test_FE[col1 + "_divide_" + col2] = X_test[col1] / X_test[col2]

for col1, col2 in combinations(ordinal_cols, 2):
    test_FE[col1 + "_multiply_" + col2] = X_test[col1] * X_test[col2]
    test_FE[col1 + "_divide_" + col2] = X_test[col1] / X_test[col2]

for col1 in num_cols:
    for col2 in ordinal_cols:
        test_FE[col1 + "_multiply_" + col2 ] = X_test[col1] * X_test[col2]
        test_FE[col1 + "_divide_" + col2 ] = X_test[col1] / X_test[col2]


X_FE_3['f1'] = (X["Number of vessels fluro"] == np.int64(3)) & (X["Max HR"] > 103.75) & (X["Sex"] == 1) #→ AUC 0.95656                                                                       
X_FE_3['f2'] = (X["Number of vessels fluro"] == np.int64(0)) & (X["ST depression"] > 1.86) & (X["Sex"] == 1)# → AUC 0.95657                                                                  
X_FE_3['f3'] = (X["Slope of ST"] == np.int64(3)) & (X["ST depression"] < 0.62) & (X["Sex"] == 1) #→ AUC 0.95658                                                                              
X_FE_3['f4'] = (X["Slope of ST"] == np.int64(3)) & (X["Cholesterol"] > 235.50) & (X["Exercise angina"] == 1)# → AUC 0.95660                                                                  
X_FE_3['f5'] = (X["Number of vessels fluro"] == np.int64(0)) & (X["BP"] < 120.50) & (X["Sex"] == 1)# → AUC 0.95660                                                                           
X_FE_3['f6'] = (X["Slope of ST"] == np.int64(2)) & (X["ST depression"] > 1.24) & (X["Sex"] == 1)# → AUC 0.95660                                                                              
X_FE_3['f7'] = (X["Slope of ST"] == np.int64(3)) & (X["Age"] < 61.00) & (X["Sex"] == 1) #→ AUC 0.95663                                                                                       
X_FE_3['f8'] = (X["Chest pain type"] == np.int64(2)) & (X["Age"] > 45.00) & (X["Sex"] == 1) #→ AUC 0.95663                                                                                   
X_FE_3['f9'] = (X["Chest pain type"] == np.int64(1)) & (X["Age"] < 45.00) & (X["Exercise angina"] == 1) #→ AUC 0.95665  


test_FE_3['f1'] = (test["Number of vessels fluro"] == np.int64(3)) & (test["Max HR"] > 103.75) & (test["Sex"] == 1) #→ AUC 0.95656                                                                       
test_FE_3['f2'] = (test["Number of vessels fluro"] == np.int64(0)) & (test["ST depression"] > 1.86) & (test["Sex"] == 1)# → AUC 0.95657                                                                  
test_FE_3['f3'] = (test["Slope of ST"] == np.int64(3)) & (test["ST depression"] < 0.62) & (test["Sex"] == 1) #→ AUC 0.95658                                                                              
test_FE_3['f4'] = (test["Slope of ST"] == np.int64(3)) & (test["Cholesterol"] > 235.50) & (test["Exercise angina"] == 1)# → AUC 0.95660                                                                  
test_FE_3['f5'] = (test["Number of vessels fluro"] == np.int64(0)) & (test["BP"] < 120.50) & (test["Sex"] == 1)# → AUC 0.95660                                                                           
test_FE_3['f6'] = (test["Slope of ST"] == np.int64(2)) & (test["ST depression"] > 1.24) & (test["Sex"] == 1)# → AUC 0.95660                                                                              
test_FE_3['f7'] = (test["Slope of ST"] == np.int64(3)) & (test["Age"] < 61.00) & (test["Sex"] == 1) #→ AUC 0.95663                                                                                       
test_FE_3['f8'] = (test["Chest pain type"] == np.int64(2)) & (test["Age"] > 45.00) & (test["Sex"] == 1) #→ AUC 0.95663                                                                                   
test_FE_3['f9'] = (test["Chest pain type"] == np.int64(1)) & (test["Age"] < 45.00) & (test["Exercise angina"] == 1) #→ AUC 0.95665  



model = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=3,
    random_state=42,
    verbosity = -1
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_FE_3, y)):

    print(f"Fold {fold+1}")
    X_train, X_valid = X_FE_3.iloc[train_idx], X_FE_3.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model.fit(X_train, y_train)
    oof_preds[valid_idx] = model.predict_proba(X_valid)[:,1]
    test_preds += model.predict_proba(test_FE_3)[:,1] / skf.n_splits

meta_X = pd.DataFrame({"lgbm_fe_3_X":oof_preds})
meta_test = pd.DataFrame({"lgbm_fe_3_test":test_preds})

meta_X.to_csv("lgbm_fe_3_X.csv",index=False)
meta_test.to_csv("lgbm_fe_3_test.csv",index=False)