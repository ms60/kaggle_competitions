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


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

print(X.head())
print(y.head())

####### elastic

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# oof_pred_elastic = np.zeros(len(X))

# for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
#     X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
#     y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

#     model = LogisticRegression(
#     penalty="elasticnet",
#     C=1.0,
#     l1_ratio=0.5,       # 0 → Ridge, 1 → Lasso
#     solver="saga",      # şart
#     max_iter=1000,
#     n_jobs=-1
#     )
#     model.fit(X_tr, y_tr)

#     # probability → [:, 1] ÇOK ÖNEMLİ
#     oof_pred_elastic[val_idx] = model.predict_proba(X_val)[:, 1]

#     fold_auc = roc_auc_score(y_val, oof_pred_elastic[val_idx])
#     print(f"Fold {fold} AUC: {fold_auc:.4f}")

# # Genel OOF skor
# print("OOF AUC:", roc_auc_score(y, oof_pred_elastic))


####### ridge

nominal_cols = ["Sex","Chest pain type","Thallium","EKG results"] # "EKG results"
num_cols = ["Age","BP","Cholesterol","Max HR","ST depression","Slope of ST","Number of vessels fluro",]
#te_cols = ["EKG results","Sex","Chest pain type","Thallium"]
#ordinal_cols = []
yes_or_no = ["FBS over 120","Exercise angina"]

preprocess = make_column_transformer(
    #(OneHotEncoder(handle_unknown="ignore") , nominal_cols),
   
    (StandardScaler() , num_cols ),
    remainder="passthrough"

)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_pred_ridge = np.zeros(len(X))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]


    pipe = Pipeline([
    ("te", TargetEncoder(cols=nominal_cols, smoothing=5) ),
    ("preprocess", preprocess),
    
    ("model", LogisticRegression(
    penalty="l2",        # Ridge
    C=1.0,               # Regularization gücü (küçük C = daha güçlü)
    solver="lbfgs",      # default ve stabil
    max_iter=1000,
    n_jobs=-1
    ))
    ])

    pipe.fit(X_tr, y_tr)

    # probability → [:, 1] ÇOK ÖNEMLİ
    oof_pred_ridge[val_idx] = pipe.predict_proba(X_val)[:, 1]

    fold_auc = roc_auc_score(y_val, oof_pred_ridge[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

# Genel OOF skor
print("OOF AUC:", roc_auc_score(y, oof_pred_ridge))


# ####### lasso

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# oof_pred_lasso = np.zeros(len(X))

# for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
#     X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
#     y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

#     model = LogisticRegression(
#     penalty="l1",        # Lasso
#     C=1.0,
#     solver="liblinear", # veya "saga"
#     max_iter=1000
#     )
#     model.fit(X_tr, y_tr)

#     # probability → [:, 1] ÇOK ÖNEMLİ
#     oof_pred_lasso[val_idx] = model.predict_proba(X_val)[:, 1]

#     fold_auc = roc_auc_score(y_val, oof_pred_lasso[val_idx])
#     print(f"Fold {fold} AUC: {fold_auc:.4f}")

# # Genel OOF skor
# print("OOF AUC:", roc_auc_score(y, oof_pred_lasso))


####### lgbm

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# oof_pred_lgbm = np.zeros(len(X))

# for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
#     X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
#     y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
#     lg_params = {'n_estimators': 4290, 'max_depth': 3, 'num_leaves': 39, 'learning_rate': 0.03530678249463069, 'min_child_samples': 48, 'subsample': 0.14215733250141216, 'colsample_bytree': 0.5462748555787986, 'reg_alpha': 0.33755662630401917, 'reg_lambda': 0.040551519519335555}
#     model = LGBMClassifier(**lg_params)
#     model.fit(X_tr, y_tr)

#     # probability → [:, 1] ÇOK ÖNEMLİ
#     oof_pred_lgbm[val_idx] = model.predict_proba(X_val)[:, 1]

#     fold_auc = roc_auc_score(y_val, oof_pred_lgbm[val_idx])
#     print(f"Fold {fold} AUC: {fold_auc:.4f}")

# # Genel OOF skor
# print("OOF AUC:", roc_auc_score(y, oof_pred_lgbm))

# X["oof_pred_elastic"] = oof_pred_elastic
X["oof_pred_ridge"] = oof_pred_ridge
# X["oof_pred_lasso"] = oof_pred_lasso
#X["oof_pred_lgbm"] = oof_pred_lgbm

X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)



def objective(trial):

    params= {
        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
        'max_depth': trial.suggest_int('max_depth', 3, 32),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'class_weight': None,  # dilersen 'balanced' da ekleyebilirsin
        'random_state': 42,
        'verbosity': -1
    }

    model = LGBMClassifier(**params)
    model.fit(X_train,y_train)

    #y_preds = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, y_proba)
    
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best ROC AUC:", study.best_value)
print("Best params:", study.best_params)

#best_params = {'n_estimators': 4290, 'max_depth': 3, 'num_leaves': 39, 'learning_rate': 0.03530678249463069, 'min_child_samples': 48, 'subsample': 0.14215733250141216, 'colsample_bytree': 0.5462748555787986, 'reg_alpha': 0.33755662630401917, 'reg_lambda': 0.040551519519335555}

# best_model = LGBMClassifier(**best_params)
# best_model.fit(X_train,y_train)
# probas = best_model.predict_proba( test.drop("id",axis=1) )[:,1]

# result = pd.DataFrame({"id":test["id"] , "Heart Disease": probas  })
# result.to_csv("result.csv",index=False)