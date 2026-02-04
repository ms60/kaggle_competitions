import optuna
import pandas as pd
import numpy as np

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

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


for i in range(1,13):
    X["ind_"+str(i)] = X["ind_"+str(i)].astype(int) 

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

for i in range(1,13):
    test["ind_"+str(i)] = test["ind_"+str(i)].astype(int) 


from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def objective(trial):
    params = {
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
    
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr ,
                  eval_set=[(X_val, y_val)],
                  eval_metric="roc_auc",
                  callbacks=[early_stopping(100),log_evaluation(0)])
        
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
    
    score = roc_auc_score(y, oof)
    return score


# study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
# study.optimize(objective, n_trials=50)

# print("Best ROC AUC:", study.best_value)
# print("Best params:", study.best_params)


# best_params = study.best_params  # Optuna’dan
best_params = {'n_estimators': 7025, 'max_depth': 22, 'num_leaves': 15, 'learning_rate': 0.012752427604733264, 'min_child_samples': 34, 'subsample': 0.6090293649529753, 'colsample_bytree': 0.12166593548807814, 'reg_alpha': 0.0038796943982102212, 'reg_lambda': 0.3101264295152337,"verbosity":-1}


oof_pred_lgbm = np.zeros(len(X))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    model = LGBMClassifier(**best_params)
    model.fit(X_tr, y_tr)
    
    oof_pred_lgbm[val_idx] = model.predict_proba(X_val)[:, 1]
    
    fold_auc = roc_auc_score(y_val, oof_pred_lgbm[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

print("OOF AUC:", roc_auc_score(y, oof_pred_lgbm))

pd.Series(oof_pred_lgbm, name="oof_pred_lgbm").to_csv("oof_pred_lgbm.csv", index=False)

# --------------------------------
# Test OOF üret
# --------------------------------
test_oof = np.zeros(len(test))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for tr_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    model = LGBMClassifier(**best_params)
    model.fit(X_tr, y_tr)

    # Fold modeli ile test tahmini
    test_oof += model.predict_proba(test.drop("id", axis=1))[:,1] / skf.n_splits

# Kaydet
pd.Series(test_oof, name="oof_pred_lgbm_test").to_csv("oof_pred_lgbm_test.csv", index=False)
print("Test OOF kaydedildi: oof_pred_lgbm_test.csv")