from itertools import combinations, product
import random
from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from matplotlib import pyplot as plt
import seaborn as sns

import shap
from tqdm import tqdm
from xgboost import XGBClassifier


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

train["gender"] = train["gender"].map({"Female": 0, "Male": 1})
test["gender"] = test["gender"].map({"Female": 0, "Male": 1})

train["Partner"] = train["Partner"].map({"No": 0, "Yes": 1})
test["Partner"] = test["Partner"].map({"No": 0, "Yes": 1})

train["Dependents"] = train["Dependents"].map({"No": 0, "Yes": 1})
test["Dependents"] = test["Dependents"].map({"No": 0, "Yes": 1})

train["PhoneService"] = train["PhoneService"].map({"No": 0, "Yes": 1})
test["PhoneService"] = test["PhoneService"].map({"No": 0, "Yes": 1})


train["MultipleLines"] = train["MultipleLines"].map({"No": 0, "Yes": 1, "No phone service": 0})
test["MultipleLines"] = test["MultipleLines"].map({"No": 0, "Yes": 1, "No phone service": 0})

train["InternetService"] = train["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2})
test["InternetService"] = test["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2})

train["OnlineSecurity"] = train["OnlineSecurity"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["OnlineSecurity"] = test["OnlineSecurity"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["OnlineBackup"] = train["OnlineBackup"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["OnlineBackup"] = test["OnlineBackup"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["DeviceProtection"] = train["DeviceProtection"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["DeviceProtection"] = test["DeviceProtection"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["TechSupport"] = train["TechSupport"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["TechSupport"] = test["TechSupport"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["StreamingTV"] = train["StreamingTV"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["StreamingTV"] = test["StreamingTV"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["StreamingMovies"] = train["StreamingMovies"].map({"No": 0, "Yes": 1, "No internet service": 0})
test["StreamingMovies"] = test["StreamingMovies"].map({"No": 0, "Yes": 1, "No internet service": 0})

train["PaperlessBilling"] = train["PaperlessBilling"].map({"No": 0, "Yes": 1})
test["PaperlessBilling"] = test["PaperlessBilling"].map({"No": 0, "Yes": 1})



#---------------------------------------------------




# #current
# train["tenure_treshold"] = (train["tenure"] > 17.0).astype(int)
# test["tenure_treshold"] = (test["tenure"] > 17.0).astype(int)

# # next

# cm = train["Contract"].map({"Month-to-month": 1, "One year": 12, "Two year": 24})
# train['Monthly_per_Contract_Month'] = train['MonthlyCharges'] / cm



#--------------------------------------------------


train["Churn"] = train["Churn"].map({"No": 0, "Yes": 1})

X = train.drop("id",axis=1)
y = X.pop("Churn")

X_test = test.drop("id",axis=1)

#------------------------------------------------


numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
ohe_features = ["Contract","PaymentMethod"]
binary_features = [col for col in X.columns if col not in numeric_features + ohe_features + ["Churn"]]


for col in ohe_features:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")




#---------------------------------------------------
# create new features

X_FE = X.copy()
X_FE_test = X_test.copy()

X_FE["tenure_squared"] = X_FE["tenure"] * X_FE["tenure"]
X_FE_test["tenure_squared"] = X_FE_test["tenure"] * X_FE_test["tenure"]


X_FE["TotalCharges_squared"] = X_FE["TotalCharges"] * X_FE["TotalCharges"]
X_FE_test["TotalCharges_squared"] = X_FE_test["TotalCharges"] * X_FE_test["TotalCharges"]


X_FE["MonthlyCharges_squared"] = X_FE["MonthlyCharges"] * X_FE["MonthlyCharges"]
X_FE_test["MonthlyCharges_squared"] = X_FE_test["MonthlyCharges"] * X_FE_test["MonthlyCharges"]

#--------------------------------------------------




 



#stats encoding
# for col in ohe_features:
#     stats = train.groupby(col)["Churn"].agg(["mean","std","count"])

#     for s in stats.columns:
#         X_FE[f"{col}_{s}"] = X_FE[col].map(stats[s]).astype("float32")
#         X_FE_test[f"{col}_{s}"] = X_FE_test[col].map(stats[s])

# # rank encoding
# for col in ohe_features:

#     counts = X_FE[col].value_counts()
#     rank_map = counts.rank(method="dense", ascending=False)

#     X_FE[f"{col}_rank"] = X_FE[col].map(rank_map).astype("int32")
#     X_FE_test[f"{col}_rank"] = X_FE_test[col].map(rank_map).astype("int32")

# print(X_FE.head())

#------
#pseudo labeling
# pseudo = pd.read_csv("./result.csv")["Churn"]

# print( pseudo[(pseudo > 0.95) | (pseudo < 0.05) ] )

# mask = (pseudo > 0.95) | (pseudo < 0.05)

# pseudo_X = X_test[mask]
# pseudo_y = (pseudo[mask] > 0.5).astype(int)

# print(pseudo_X)
# print(pseudo_y)

# X_FE = pd.concat([X_FE, pseudo_X])
# y_FE = pd.concat([y, pseudo_y])
#---------

# X_FE['f1'] = (X_FE["Contract"] == 'One year') & (X_FE["TotalCharges"] > 885.40)
# X_FE_test['f1'] = (X_FE_test["Contract"] == 'One year') & (X_FE_test["TotalCharges"] > 885.40)

# X_FE['f2'] = (X_FE["Contract"] == 'Month-to-month') & (X_FE["MonthlyCharges"] < 28.30)
# X_FE_test['f2'] = (X_FE_test["Contract"] == 'Month-to-month') & (X_FE_test["MonthlyCharges"] < 28.30)


# X_FE['f3'] = (X_FE["Contract"] == 'Month-to-month') & (X_FE["TotalCharges"] > 4351.80)
# X_FE_test['f3'] = (X_FE_test["Contract"] == 'Month-to-month') & (X_FE_test["TotalCharges"] > 4351.80)


# X_FE['f4'] = (X_FE["PaymentMethod"] == 'Credit card (automatic)') & (X_FE["tenure"] < 15.20)
# X_FE_test['f4'] = (X_FE_test["PaymentMethod"] == 'Credit card (automatic)') & (X_FE_test["tenure"] < 15.20)


# X_FE['f5'] = (X_FE["PaymentMethod"] == 'Mailed check') & (X_FE["TotalCharges"] < 3485.20)
# X_FE_test['f5'] = (X_FE_test["PaymentMethod"] == 'Mailed check') & (X_FE_test["TotalCharges"] < 3485.20)





#-----------------------------------------------


#------------------------------------------------





#60 , 0.01
xgb_params = {
    'n_estimators': 80000,      
    'learning_rate': 0.009,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree':0.8,
    'max_bin':16000,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    #'early_stopping_rounds': 200,
    'device': 'cuda',
    
    'enable_categorical': True,
}




X_small, _, y_small, _ = train_test_split(X, y, train_size=200_000, random_state=42, stratify=y)



X_train , X_valid , y_train , y_valid = train_test_split(X_FE , y , test_size=0.075 , random_state=42 , stratify=y)
X_train_2, X_valid_2, y_train_2, y_valid_2 = train_test_split(X_FE, y, test_size=0.1, random_state=60, stratify=y)
X_train_3, X_valid_3, y_train_3, y_valid_3 = train_test_split(X_FE, y, test_size=0.2, random_state=31, stratify=y)



#------------------

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

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_work,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=0)

    base_auc = roc_auc_score(
        y_val,
        model.predict_proba(X_val)[:, 1]
    )
    
    # base_auc = cross_val_score(
    #     model,
    #     X_work, y,
    #     cv=skf,
    #     scoring="roc_auc",
    #     n_jobs=-1
    # ).min()

    print(f"Initial AUC: {base_auc:.5f}")

    accepted = {}

    for name, (col, expr) in tqdm(
        candidate_features.items(),
        total=len(candidate_features),
        desc="Evaluating feature combinations"
    ):
        tqdm.write(f"Trying: {name}")   # anlık candidate

        X_work[name] = col.astype("int8")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_work,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=0)

        auc = roc_auc_score(
            y_val,
            model.predict_proba(X_val)[:, 1]
        )

        # auc = cross_val_score(
        #     model,
        #     X_work, y,
        #     cv=skf,
        #     scoring="roc_auc",
        #     n_jobs=-1
        # ).min()

        if auc > base_auc:
            base_auc = auc
            accepted[name] = (col, expr)
            tqdm.write(f"[KEEP] X['f1'] = {expr} → AUC {auc:.5f}")
        else:
            X_work.drop(columns=[name], inplace=True)

    print(f"\nSelected {len(accepted)} features")
    print(f"Final AUC: {base_auc:.5f}")

    return accepted, X_work



#--------------------------------



# model = XGBClassifier(**xgb_params)

# cat_x =  generate_category_equality(X, ohe_features)
# #cat_x_comb = generate_cross_combinations_same(cat_x)

# num_x = generate_gt_lt_num(X,["tenure","MonthlyCharges","TotalCharges"] , [10 , 10 , 10] )
# # num_x_comb = generate_cross_combinations_same(num_x)

# cat_num_comb = generate_cross_combinations_different(cat_x, num_x)


# items = list(cat_num_comb.items())
# random.shuffle(items)

# cat_num_comb_shuffled = dict(items)


# accepted, X_final = greedy_auc_feature_addition(
#     X_small,
#     y_small,
#     cat_num_comb_shuffled,
#     model
# )


#--------------------

# xgb_params_best_2 = dict(
#     n_estimators=100_000,
#     learning_rate=0.1,
#     max_depth=3,
#     min_child_weight=5,
#     subsample=0.85,
#     colsample_bytree=0.85,
#     objective="binary:logistic",
#     eval_metric="auc",
#     tree_method="hist",
#     enable_categorical=True,
#     random_state=42,
#     n_jobs=-1,
#     early_stopping_rounds=200,
#     device="cuda",
#     #max_bin=16000,
# )
#-----

# model = XGBClassifier(**xgb_params)
# model.fit(X_train,y_train,
#           eval_set=[(X_valid, y_valid)],
#           verbose=1000
# )



# model2 = XGBClassifier(**xgb_params)
# model2.fit(X_train_2,y_train_2,
#            eval_set=[(X_valid_2, y_valid_2)],
#            verbose=1000)

# model3 = XGBClassifier(**xgb_params)
# model3.fit(X_train_3,y_train_3,
#            eval_set=[(X_valid_3, y_valid_3)],
#            verbose=1000)

# y_proba = model.predict_proba(X_valid)[:, 1]
# y_proba2 = model2.predict_proba(X_valid_2)[:, 1]
# y_proba_3 = model3.predict_proba(X_valid_3)[:, 1]



# score = roc_auc_score(y_valid, y_proba)
# score2 = roc_auc_score(y_valid_2, y_proba2)
# score3 = roc_auc_score(y_valid_3, y_proba_3)

# scores = [score, score2, score3]


# print(scores)
# print(np.mean(scores))
# print(np.std(scores))

# print("DIFF : ", np.mean(scores) - 0.9180853749696384 )


#--------------------------------------------------

lgbm_params = {
    'n_estimators': 60000,
    'learning_rate': 0.01,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree':0.8,
    'max_bin':16000,
    'objective': 'binary',
    'metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    #'early_stopping_rounds': 200,
    #'device': 'cuda',
    'verbosity':-1,
}

model = LGBMClassifier(**lgbm_params)
model.fit(X,y)

y_proba = model.predict_proba(X_test)[:, 1]

result = pd.DataFrame({
    "id": test["id"],
    "Churn": y_proba
})

result.to_csv("result.csv", index=False)

#clip

# rslt = pd.read_csv("result.csv")
#rslt["Churn"] = rslt["Churn"].apply(lambda x : 0.0 if x <= 0.01 else x)

# #Logit Scaling
# from scipy.special import logit, expit
# alpha = 1.3
# rslt["Churn"]  = expit(alpha * logit(rslt["Churn"] ))

# Power Transform for prediction sharpening
# gamma = 3
# rslt["Churn"]  = rslt["Churn"] **gamma / (rslt["Churn"] **gamma + (1-rslt["Churn"] )**gamma)
# rslt.to_csv("result_clipped.csv", index=False)


#----------------------------------------------------
#iterative pseudo-labeling

# max_iter = 5
# min_new_samples = 100

# X_train_full = X.copy()
# y_train_full = y.copy()
# X_test_remaining = X_test.copy()

# for i in range(max_iter):
#     model = XGBClassifier(**xgb_params)
#     model.fit(X_train_full, y_train_full,
#               #eval_set=[(X_train_full, y_train_full)],
#               verbose=0
#     )
#     cv_score = cross_val_score(model, X_train_full, y_train_full,
#                                cv=3, scoring="roc_auc" , n_jobs=-1)
#     print("cv auc:")
#     print(cv_score.mean() , cv_score.std())

#     probs = model.predict_proba(X_test_remaining)[:,1]

#     mask = (probs > 0.99) | (probs < 0.01)

#     pseudo_X = X_test_remaining[mask]
#     pseudo_y = (probs[mask] > 0.5).astype(int)

#     print(f"iteration {i+1} pseudo samples:", len(pseudo_X))

#     if len(pseudo_X) < min_new_samples:
#         break

#     X_train_full = pd.concat([X_train_full, pseudo_X])
#     y_train_full = np.concatenate([y_train_full, pseudo_y])

#     X_test_remaining = X_test_remaining[~mask]


#--------------------------------------------------

# def objective(trial):
#     xgb_params = {
#         "n_estimators": trial.suggest_int("n_estimators", 18000, 30000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
#         "max_depth": trial.suggest_int("max_depth", 3, 6),
#         "subsample": trial.suggest_float("subsample", 0.7, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
#         "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
#         "max_bin": trial.suggest_int("max_bin", 15000, 25000, log=True),

#         "max_depth":3,
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',
#         'n_jobs': -1,
#         'random_state': 42,
#         'early_stopping_rounds': 200,
#         'device': 'cuda',
#         'enable_categorical': True,
#     }

#     model = XGBClassifier(**xgb_params)
#     model.fit(X_train,y_train,
#             eval_set=[(X_valid, y_valid)],
#             verbose=5000
#     )



#     model2 = XGBClassifier(**xgb_params)
#     model2.fit(X_train_2,y_train_2,
#             eval_set=[(X_valid_2, y_valid_2)],
#             verbose=5000)

#     y_proba = model.predict_proba(X_valid)[:, 1]
#     y_proba2 = model2.predict_proba(X_valid_2)[:, 1]

#     score = roc_auc_score(y_valid, y_proba)
#     score2 = roc_auc_score(y_valid_2, y_proba2)

#     return (score + score2)/2


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=60)

# print("Best ROC AUC:", study.best_value)
# print("Best params:", study.best_params)


## raw baseline 0.9188629324118204
## raw baseline x_small 200_000 0.9145626863770953