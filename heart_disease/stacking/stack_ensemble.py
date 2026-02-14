import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

binary_cols = ["Exercise angina","FBS over 120"]
ordinal_cols = ["EKG results","Number of vessels fluro"]
nominal_cols = ["Sex","Chest pain type","Thallium"] 
numerical_cols = ["Age","BP","Cholesterol","Max HR","ST depression","Slope of ST"]

categorical_cols_catboost = [
    "EKG results","Thallium","Chest pain type",
    "Slope of ST","Number of vessels fluro",
    "Exercise angina","Sex","FBS over 120"
]




# =======================
# LOAD DATA
# =======================

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

X = train.drop("id", axis=1)
y = X.pop("Heart Disease")
y = y.map({"Presence": 1, "Absence": 0})

print(X.head())



X_test = test.drop("id", axis=1)


# =======================
# FEATURE GROUPS
# =======================
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


# =======================
# PREPROCESSOR
# =======================

preprocess = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), nominal_cols),
    (StandardScaler(), numerical_cols + ordinal_cols ),
    remainder="passthrough"
)


# =======================
# CV
# =======================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =======================
# MODELS
# =======================
lgbm_params1 = {'boosting_type': 'gbdt', 'n_estimators': 8000, 'learning_rate': 0.030752124591604243, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.620293411878579, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}
lgbm_params1.update({
    "objective": "binary",
    "metric": "auc",
    'verbose':-1
})

lgbm_model1 = LGBMClassifier(**lgbm_params1)




lgbm_params2 = {'boosting_type': 'gbdt', 'n_estimators': 3000, 'learning_rate': 0.02, 'num_leaves': 31, 'max_depth': 4, 'min_child_samples': 80, 'subsample': 0.8, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 0.5}
lgbm_params2.update({
    "objective": "binary",
    "metric": "auc",
    'verbose':-1
})

lgbm_model2 = LGBMClassifier(**lgbm_params2)



logreg_elastic_model = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,
    C=0.7,
    max_iter=3000,
    n_jobs=-1
)



# =======================
# OOF ARRAYS
# =======================

models = {
"lgbm_1":lgbm_model1,
"lgbm_2":lgbm_model2,
"logreg_elasticnet":logreg_elastic_model
}

oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}


# =======================
# OOF LOOP
# =======================

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

    print(f"Fold {fold+1}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Preprocess for non-tree models
    X_train_proc = preprocess.fit_transform(X_train)
    X_valid_proc = preprocess.transform(X_valid)
    X_test_proc = preprocess.transform(X_test)

    for name, model in models.items():

        if name in ["lgbm", "xgb"]:
            model.fit(X_train, y_train)
            oof_preds[name][valid_idx] = model.predict_proba(X_valid)[:,1]
            test_preds[name] += model.predict_proba(X_test)[:,1] / skf.n_splits

        elif name == "catboost":
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                cat_features=categorical_cols_catboost
            )
            oof_preds[name][valid_idx] = model.predict_proba(X_valid)[:,1]
            test_preds[name] += model.predict_proba(X_test)[:,1] / skf.n_splits

        else:
            model.fit(X_train_proc, y_train)
            oof_preds[name][valid_idx] = model.predict_proba(X_valid_proc)[:,1]
            test_preds[name] += model.predict_proba(X_test_proc)[:,1] / skf.n_splits


# =======================
# META DATA
# =======================

meta_X = pd.DataFrame(oof_preds)
meta_test = pd.DataFrame(test_preds)

meta_X_name = ""
for name in models.keys():
    meta_X_name += name + '_'

meta_X.to_csv(f"meta_X_{meta_X_name}.csv", index=False)
meta_test.to_csv(f"meta_test_{meta_X_name}.csv", index=False)


# =======================
# META MODEL CV
# =======================

# meta_model = LGBMClassifier(random_state=42)

# scores = cross_val_score(
#     meta_model,
#     meta_X,
#     y,
#     cv=skf,
#     scoring="roc_auc",
#     n_jobs=-1
# )

# print("\nMeta CV AUC scores:", scores)
# print("Meta CV AUC mean:", scores.mean())


# # =======================
# # CORRELATION CHECK
# # =======================

# print("\nOOF Correlation Matrix:")
# print(meta_X.corr().round(2))
