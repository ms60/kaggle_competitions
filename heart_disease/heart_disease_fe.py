from catboost import CatBoostClassifier
import lightgbm
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from sklearn import clone
from sklearn.base import clone
from sklearn.compose import make_column_transformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold , StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import mean_absolute_error, r2_score , accuracy_score , classification_report ,  precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder, RobustScaler , StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier 
from category_encoders import TargetEncoder

from itertools import combinations

import xgboost


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

# print(X.head())
# print(y.head())


#train fe

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

X["ind_12"] =   (X["ST depression"] > 2.5 ) & ( X["Slope of ST"] >= 2 ) & (X["Age"] > 50   )

###

# X["age_x_chol"] = X["Age"] * X["Cholesterol"]
# X["bp_x_hr"] = X["BP"] * X["Max HR"]

# X["chol_hr_ratio"] = X["Cholesterol"] / (X["Max HR"] + 1)
# X["bp_age_ratio"] = X["BP"] / (X["Age"] + 1)

# X["log_chol"] = np.log1p(X["Cholesterol"])
# X["sqrt_bp"] = np.sqrt(X["BP"])

# X["age_risk"] = pd.cut(
#     X["Age"],
#     bins=[0, 40, 55, 70, 100],
#     labels=[0,1,2,3]
# ).astype(int)

# X["num_mean"] = X[num_cols].mean(axis=1)
# X["num_std"]  = X[num_cols].std(axis=1)
# X["num_max"]  = X[num_cols].max(axis=1)


for i in range(1,13):
    X["ind_"+str(i)] = X["ind_"+str(i)].astype(int) 

#test fe 

test["ind_1"] = (test["Age"] > 40 ) & ( test["Exercise angina"] )
test["ind_2"] = (test["Age"] > 50) &  (test["FBS over 120"] )
test["ind_3"] = ( test["Exercise angina"] ) &  (test["ST depression"] > 1 )
test["ind_4"] = ( test["Exercise angina"] ) & (test["FBS over 120"] )

test["ind_5"] = (test["ind_4"] ) & ( test["EKG results"] > 0 )

test["ind_6"] = test["ind_3"] &  test["ind_4"] & (test["Thallium"] == 3)

test["ind_7"] = ( test["EKG results"] > 0 ) & ( test["Chest pain type"]==3 )
test["ind_8"] = ( test["EKG results"] > 0 ) & ( test["Chest pain type"]==2 )

test["ind_9"] =  ( test["Number of vessels fluro"] > 0 ) & (test["Age"] < 40) 

test["ind_10"] =  (test["Sex"] ==1 )  & (test["Age"] > 40 ) & ( test["Cholesterol"] > 300 )
test["ind_11"] =  (test["Sex"] ==1 )  & (test["Age"] > 40 ) & ( test["BP"] > 180 )

test["ind_12"] =   (test["ST depression"] > 2.5 ) & ( test["Slope of ST"] >= 2 ) & (test["Age"] > 50   )


# test["age_x_chol"] = test["Age"] * test["Cholesterol"]
# test["bp_x_hr"] = test["BP"] * test["Max HR"]

# test["chol_hr_ratio"] = test["Cholesterol"] / (test["Max HR"] + 1)
# test["bp_age_ratio"] = test["BP"] / (test["Age"] + 1)

# test["log_chol"] = np.log1p(test["Cholesterol"])
# test["sqrt_bp"] = np.sqrt(test["BP"])

# test["age_risk"] = pd.cut(
#     test["Age"],
#     bins=[0, 40, 55, 70, 100],
#     labels=[0,1,2,3]
# ).astype(int)

for i in range(1,13):
    test["ind_"+str(i)] = test["ind_"+str(i)].astype(int) 


# Age > 55 AND Exercise_angina
# Max_HR < 120 AND ST_depression > 2


print(X.head())
# print(X["ST depression"].value_counts())
# print(X["Slope of ST"].value_counts())





nominal_cols = ["Sex","Chest pain type","Thallium"] 
num_cols = ["Age","BP","Cholesterol","Max HR","ST depression","Slope of ST"]
ordinal_cols = ["EKG results","Number of vessels fluro" ]
yes_or_no = ["FBS over 120","Exercise angina"]

# ordinal_cols.append("age_risk")
# num_cols.append("age_x_chol")
# num_cols.append("bp_x_hr")

# num_cols.append("chol_hr_ratio")
# num_cols.append("bp_age_ratio")

# num_cols.append("log_chol")
# num_cols.append("sqrt_bp")

num_all_cols = num_cols + ordinal_cols

for i in range(1,13):
    yes_or_no.append( X["ind_"+str(i)] ) 




preprocess = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , nominal_cols ),
    (StandardScaler() , num_all_cols ),
    remainder="passthrough"

)

preprocess_robust = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , nominal_cols ),
    (RobustScaler() , num_all_cols ),
    #(FunctionTransformer(lambda x: x), yes_or_no),
    remainder="drop"

)


############################

    # "ridge_all": LogisticRegression(penalty="l2"),
    # "knn_local": KNeighborsClassifier(n_neighbors=25),
    # "nb_dist": GaussianNB(),
    # "svm_margin": LinearSVC(),
    # "mlp_smooth": MLPClassifier(hidden_layer_sizes=(64, 32)),
    # "xgb_rules": XGBClassifier(max_depth=4)


# Models
svc = SVC(kernel='linear', class_weight='balanced',random_state=42)
rfc = RandomForestClassifier(random_state=42)
knc = KNeighborsClassifier()
gbc = GradientBoostingClassifier(random_state=42)
xgb = xgboost.XGBClassifier(objective='binary:logistic', enable_categorical=True, device='cuda',random_state=42, eval_metric="auc")
ada = AdaBoostClassifier(random_state=42)
hgbc = HistGradientBoostingClassifier(scoring='roc_auc', class_weight='balanced', random_state=42)
lgbm = lightgbm.LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, random_state=42,device ='cpu', verbosity=-1)
catc = CatBoostClassifier(loss_function='Logloss' , eval_metric='AUC', auto_class_weights='Balanced', random_state=123,task_type='CPU' , verbose=False)
sgdc = SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42) , 
lgstc_ridge = LogisticRegression(penalty="l2",C=1.0,solver="lbfgs",max_iter=1000,random_state=42)
lgstc_elasticnet = LogisticRegression(penalty="elasticnet",l1_ratio=0.2,solver="saga",C=1.0,max_iter=2000,random_state=42)
lgstc_lasso = LogisticRegression(penalty="l1",C=1.0,solver="liblinear",max_iter=1000,random_state=42)


non_tree_models = {
    # "svc": Pipeline([
    #     ("prep", preprocess),
    #     ("model", SVC(
    #         kernel="linear",
    #         probability=True,
    #         class_weight="balanced",
    #         random_state=42
    #     ))
    # ]),
    
    "sgdc": Pipeline([
        ("prep", preprocess),
        ("model", SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "lgstc_all": Pipeline([
        ("prep", preprocess),
        ("model", LogisticRegression(penalty="l2"))
    ]),

    "knn_local": Pipeline([
        ("prep", preprocess_robust),
        ("model", KNeighborsClassifier(n_neighbors=25))
    ]),

    "mlp": Pipeline([
        ("prep", preprocess),
        ("model", MLPClassifier(hidden_layer_sizes=(64, 32)))
    ]),

    "nb_dist": Pipeline([
        ("prep", preprocess),
        ("model", GaussianNB())
    ]),

    "lgstc_ridge": Pipeline([
        ("prep", preprocess),
        ("model", lgstc_ridge)
    ]),
    "lgstc_elasticnet":Pipeline([
        ("prep",preprocess),
        ("model",lgstc_elasticnet)
    ]),

    "lgstc_lasso":Pipeline([
        ("prep",preprocess),
        ("model",lgstc_lasso)
    ])
}

tree_models = {
    "xgb": xgb,
    "lgbm": lgbm,
    "catc": catc,
    "rfc": rfc,
    "hgbc": hgbc,
    "gbc": gbc,
    "ada": ada
}

#models = {"svc":svc , "rfc":rfc , "knc":knc , "gbc":gbc, "xgb":xgb ,"ada":ada , "hgbc":hgbc , "lgbm":lgbm , "catc":catc , "sgdc":sgdc , "lgstc_ridge":lgstc_ridge , "lgstc_elasticnet":lgstc_elasticnet , "lgstc_lasso":lgstc_lasso }

models = {**non_tree_models, **tree_models}

model_oofs = {name: np.zeros(len(X)) for name in models}

##############################


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    for model_name, base_model in models.items():
        print(f"{model_name} | fold {fold}" )
        model = clone(base_model)
        model.fit(X_tr, y_tr)

        oof_pred = model.predict_proba(X_val)[:, 1]
        model_oofs[model_name][val_idx] = oof_pred

        auc = roc_auc_score(y_val, oof_pred)
        print(f"{model_name} | fold {fold} | AUC {auc:.4f}")



oof_df = pd.DataFrame(model_oofs)
corr = oof_df.corr()
print(corr)

selected_models = models

# selected_models = {
#     "lgbm": models["lgbm"],
#     "lgstc_ridge": models["lgstc_ridge"],   # pipeline’lı
#     "knn_local": models["knn_local"],
#     "ada": models["ada"],
#     "nb_dist":models["nb_dist"]
# }

n_folds = skf.n_splits

oof_preds = {m: np.zeros(len(X)) for m in selected_models}
test_preds = {m: np.zeros(len(test.drop("id",axis=1))) for m in selected_models}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFOLD {fold}")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    for model_name, base_model in selected_models.items():
        model = clone(base_model)
        model.fit(X_tr, y_tr)

        # ---- OOF ----
        oof_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[model_name][val_idx] = oof_pred

        # ---- TEST ----
        test_pred = model.predict_proba(test.drop("id",axis=1))[:, 1]
        test_preds[model_name] += test_pred / n_folds

        fold_auc = roc_auc_score(y_val, oof_pred)
        print(f"{model_name:12s} | fold AUC = {fold_auc:.4f}")


for m in oof_preds:
    auc = roc_auc_score(y, oof_preds[m])
    print(f"{m:12s} OOF AUC = {auc:.4f}")







X_meta_train = pd.DataFrame(oof_preds)
X_meta_test = pd.DataFrame(test_preds)

X_meta_train.to_csv("./data/oof_train.csv",index=False)
X_meta_test.to_csv("./data/oof_test.csv",index=False)


# X_meta = X_meta_train.values
# y_meta = y.values

# def objective(trial):
#     alpha = trial.suggest_float("alpha", 1e-4, 100, log=True)
    
#     meta = Ridge(alpha=alpha, random_state=42)

#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     aucs = []

#     for tr_idx, val_idx in cv.split(X_meta, y_meta):
#         X_tr, X_val = X_meta[tr_idx], X_meta[val_idx]
#         y_tr, y_val = y_meta[tr_idx], y_meta[val_idx]

#         meta.fit(X_tr, y_tr)
#         preds = meta.predict(X_val)
#         aucs.append(roc_auc_score(y_val, preds))

#     return np.mean(aucs)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# print("Best alpha:", study.best_params["alpha"])
# print("Best AUC:", study.best_value)



# meta = Ridge(alpha=study.best_params["alpha"])
# meta.fit(X_meta_train, y)

# test_meta_pred = meta.predict(X_meta_test)

# for name, coef in zip(X_meta_train.columns, meta.coef_):
#     print(f"{name:12s} weight = {coef:.4f}")


# submission = pd.DataFrame({
#     "id": test["id"],
#     "Heart Disease": test_meta_pred
# })

# submission.to_csv("ridge_meta_submission.csv", index=False)
#######





