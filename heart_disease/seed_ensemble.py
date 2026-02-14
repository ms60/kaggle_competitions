from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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

best_params = {'boosting_type': 'gbdt', 'n_estimators': 6405, 'learning_rate': 0.030752124591604243, 'num_leaves': 75, 'max_depth': 3, 'min_child_samples': 178, 'subsample': 0.620293411878579, 'colsample_bytree': 0.13455421264459272, 'reg_alpha': 3.924444416649399, 'reg_lambda': 0.26152458198337813}

best_params.update({
    #'n_estimators': 8000,
    "objective": "binary",
    "metric": "auc",
    'verbose':-1
})


#seeds = [42, 52, 62, 72, 82]
seeds = [i for i in range(100)]
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof = np.zeros(len(X))

#-------------------------------------------------

# cv score part

# for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
#     print(f"Fold {fold+1}")    
#     fold_preds = np.zeros(len(valid_idx))
    
#     for seed in seeds:
#         model = LGBMClassifier(**best_params, random_state=seed)
#         model.fit(X.iloc[train_idx], y.iloc[train_idx])
        
#         fold_preds += model.predict_proba(X.iloc[valid_idx])[:, 1]
    
#     fold_preds /= len(seeds)
#     oof[valid_idx] = fold_preds

# auc = roc_auc_score(y, oof)
# print("Seed Ensemble CV AUC:", auc)

# prediction part

# test_preds = np.zeros(len(X_test))

# for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    
#     X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    
#     for seed in seeds:
        
#         model = LGBMClassifier(
#             **best_params,
#             random_state=seed
#         )
        
#         model.fit(X_train, y_train)
        
#         test_preds += model.predict_proba(X_test)[:,1]

# # Ortalama
# test_preds /= (len(seeds) * skf.n_splits)

# result = pd.DataFrame({"id":test["id"] , "Heart Disease":test_preds })
# result.to_csv("seed_ensemble.csv",index=False)

#------------------------------------------------------


# 1️⃣ Her seed için OOF üret
oof_dict = {}

for seed in seeds:
    
    oof = np.zeros(len(X))
    
    for train_idx, valid_idx in skf.split(X, y):
        
        model = LGBMClassifier(
            **best_params,
            random_state=seed
        )
        
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    
    oof_dict[seed] = oof
    
    print(f"Seed {seed} CV:", roc_auc_score(y, oof))

# 2️⃣ Greedy selection başlat
selected_seeds = []
remaining_seeds = seeds.copy()

best_score = 0
ensemble_oof = np.zeros(len(X))

while len(remaining_seeds) > 0:
    
    scores = []
    
    for seed in remaining_seeds:
        
        temp_oof = (
            ensemble_oof * len(selected_seeds) + oof_dict[seed]
        ) / (len(selected_seeds) + 1)
        
        score = roc_auc_score(y, temp_oof)
        scores.append((score, seed))
    
    scores.sort(reverse=True)
    best_new_score, best_seed = scores[0]
    
    if best_new_score > best_score:
        
        selected_seeds.append(best_seed)
        remaining_seeds.remove(best_seed)
        
        ensemble_oof = (
            ensemble_oof * (len(selected_seeds)-1) + oof_dict[best_seed]
        ) / len(selected_seeds)
        
        best_score = best_new_score
        
        print(f"Added seed {best_seed} → New CV: {best_score}")
    
    else:
        print("No improvement, stopping.")
        break

print("Final selected seeds:", selected_seeds)
print("Final CV:", best_score)


test_preds = np.zeros(len(X_test))

for seed in selected_seeds:
    
    model = LGBMClassifier(**best_params, random_state=seed)
    model.fit(X, y)
    
    test_preds += model.predict_proba(X_test)[:,1]

test_preds /= len(selected_seeds)

result = pd.DataFrame({"id":test["id"] , "Heart Disease":test_preds })
result.to_csv("seed_ensemble.csv",index=False)
