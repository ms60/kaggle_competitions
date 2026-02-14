from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

X_test = test.drop("id", axis=1)

X["f1"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Sex"] == 1)
X["f2"] = (X["Age"] < 37.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f3"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["Exercise angina"] == 1)
X["f4"] = (X["Age"] > 69.00) & (X["Thallium"] == 7) & (X["FBS over 120"] == 1)

X["f5"] =   (X["Sex"] == 1) & (X["Chest pain type"] == 3) & (X["EKG results"] == 2)

X["f6"] = (X["Thallium"] == 3) & (X["Age"] < 53.00)

for i in range(1,7):
    X["f"+str(i)] = X["f"+str(i)].astype(int) 

X_test["f1"] = (X_test["Age"] < 37.00) & (X_test["Thallium"] == 7) & (X_test["Sex"] == 1)
X_test["f2"] = (X_test["Age"] < 37.00) & (X_test["Thallium"] == 7) & (X_test["Exercise angina"] == 1)
X_test["f3"] = (X_test["Age"] > 69.00) & (X_test["Thallium"] == 7) & (X_test["Exercise angina"] == 1)
X_test["f4"] = (X_test["Age"] > 69.00) & (X_test["Thallium"] == 7) & (X_test["FBS over 120"] == 1)

X_test["f5"] =   (X_test["Sex"] == 1) & (X_test["Chest pain type"] == 3) & (X_test["EKG results"] == 2)

X_test["f6"] = (X_test["Thallium"] == 3) & (X_test["Age"] < 53.00)

for i in range(1,7):
    X_test["f"+str(i)] = X_test["f"+str(i)].astype(int) 

best_params_0 = { 'verbosity':-1 ,'boosting_type': 'gbdt', 'n_estimators': 6405, 'learning_rate': 0.030752124591604243, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.620293411878579, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}

best_params_1 = {'verbosity':-1 ,'boosting_type': 'gbdt', 'n_estimators': 6800, 'learning_rate': 0.028, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.60, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}

best_params_2 = {'verbosity':-1 ,'boosting_type': 'gbdt', 'n_estimators': 6405, 'learning_rate': 0.030752124591604243, 'num_leaves': 95, 'max_depth': 4, 'min_child_samples': 150, 'subsample': 0.620293411878579, 'colsample_bytree': 0.2, 'reg_alpha': 3.0, 'reg_lambda': 0.26152458198337813}

best_params_3 = {'verbosity':-1 ,'boosting_type': 'gbdt', 'n_estimators': 6405, 'learning_rate': 0.030752124591604243, 'num_leaves': 60, 'max_depth': 3, 'min_child_samples': 210, 'subsample': 0.620293411878579, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 4.5, 'reg_lambda': 0.5}

best_params_4 = {'verbosity':-1 ,'random_state': 99,'boosting_type': 'gbdt', 'n_estimators': 6405, 'learning_rate': 0.030752124591604243, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.70, 'colsample_bytree': 0.3, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}

best_params_5 = { 'verbosity':-1 ,'boosting_type': 'gbdt', 'n_estimators': 12000, 'learning_rate': 0.030752124591604243, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.620293411878579, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}
#"max_bin":63,"force_col_wise":True,

model_0 = LGBMClassifier(**best_params_0)
model_1 = LGBMClassifier(**best_params_1)
model_2 = LGBMClassifier(**best_params_2)
model_3 = LGBMClassifier(**best_params_3)
model_4 = LGBMClassifier(**best_params_4)
model_5 = LGBMClassifier(**best_params_5)


# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=60)
# scores = cross_val_score(
#     model_5,
#     X, y,
#     cv=cv,
#     scoring="roc_auc",
#     n_jobs=-1
# )
# print(scores)
# print(scores.mean())
# print(scores.std())

best_model = LGBMClassifier(**best_params_5)



best_model.fit(X,y)
test_preds = best_model.predict_proba(X_test)[:,1]

result = pd.DataFrame({"id":test["id"] , "Heart Disease":test_preds })
result.to_csv("param_pert.csv",index=False)
