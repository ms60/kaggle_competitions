from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import arff

from sklearn.compose import make_column_transformer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score , accuracy_score 
from sklearn.model_selection import StratifiedKFold, train_test_split

pd.set_option('display.max_columns', None)

data, meta = arff.loadarff("./data/dataset_57_hypothyroid.arff")

dataset = pd.DataFrame(data)

# byte string → normal string dönüşümü (gerekirse)
for col in dataset.select_dtypes([object]):
    dataset[col] = dataset[col].str.decode("utf-8")

# print(dataset.head())
# print(dataset.shape)

# print(dataset["Class"].value_counts() )

# print(dataset.isnull().sum())

# print(dataset.dtypes)

# print("-"*80)

#-------------------

binary_cols = ["on_thyroxine","query_on_thyroxine","on_antithyroid_medication","sick","pregnant","thyroid_surgery","I131_treatment","query_hypothyroid","query_hyperthyroid","lithium","goitre","tumor","hypopituitary","psych","TSH_measured","T3_measured","TT4_measured","T4U_measured","FTI_measured","TBG_measured"]
numeric_cols = ["age","TSH","T3","TT4","T4U","FTI"]
ohe_cols = ["referral_source"]



for col in binary_cols:
   dataset[col] = dataset[col].map({"f": 0, "t": 1})

dataset["sex"] = dataset["sex"].map({"F": 0, "M": 1})

dataset = dataset.drop("TBG",axis=1) # no data drop TBG
dataset["age"] = dataset["age"].fillna(dataset["age"].median()) # 1 age missing fill with median

dataset_hypo = dataset[ dataset["Class"] != "negative" ]
dataset["Class"] = dataset["Class"].apply(lambda x : 0 if x=='negative' else 1)



# print(dataset.head())
# print(dataset["Class"].value_counts())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# print(dataset[numeric_cols].describe().T )




#-----------
#HOLDOUT

X = dataset.drop(columns=["Class"])
y = dataset["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#----------------
for col in ["age","TSH","T3","TT4","T4U","FTI"]:
    lower = X_train[col].quantile(0.01)
    upper = X_train[col].quantile(0.99)
    X_train[col] = X_train[col].clip(lower, upper)
    X_test[col] = X_test[col].clip(lower, upper)

#----------------
# missing cols: TSH , T3 , TT4 , T4U , FTI 


sex_mode = X_train["sex"].mode()[0]
tsh_median = X_train["TSH"].median()
t3_median = X_train["T3"].median()
tt4_median = X_train["TT4"].median()
t4u_median = X_train["T4U"].median()
fti_median = X_train["FTI"].median()

for col in ["sex", "TSH", "T3", "TT4", "T4U", "FTI"]:
    X_train[f"{col}_isnull"] = X_train[col].isnull().astype(int)
    X_test[f"{col}_isnull"] = X_test[col].isnull().astype(int)

X_train["T4U_FTI_isnull"] = X_train["T4U"].isnull() & X_train["FTI"].isnull()
X_test["T4U_FTI_isnull"] = X_test["T4U"].isnull() & X_test["FTI"].isnull()

X_train["T4U_FTI_TSH_isnull"] = X_train["T4U"].isnull() & X_train["FTI"].isnull() & X_train["TSH"].isnull()
X_test["T4U_FTI_TSH_isnull"] = X_test["T4U"].isnull() & X_test["FTI"].isnull() & X_test["TSH"].isnull()



# X_train["sex_ref_missing"] = (X_train["sex"].isnull()) & (X_train["referral_source"].isnull()).astype(int)
# X_test["sex_ref_missing"] = (X_test["sex"].isnull()) & (X_test["referral_source"].isnull()).astype(int)

# X_train["FTI_T4U_missing"] = (X_train["FTI"].isnull()) & (X_train["T4U"].isnull() & X_train["TSH"].isnull()).astype(int)
# X_test["FTI_T4U_missing"] = (X_test["FTI"].isnull()) & (X_test["T4U"].isnull() & X_test["TSH"].isnull()).astype(int)



X_train["sex"] = X_train["sex"].fillna(sex_mode)
X_train["TSH"] = X_train["TSH"].fillna(tsh_median)
X_train["T3"] = X_train["T3"].fillna(t3_median)
X_train["TT4"] = X_train["TT4"].fillna(tt4_median)
X_train["T4U"] = X_train["T4U"].fillna(t4u_median)
X_train["FTI"] = X_train["FTI"].fillna(fti_median)

X_test["sex"] = X_test["sex"].fillna(sex_mode)
X_test["TSH"] = X_test["TSH"].fillna(tsh_median)
X_test["T3"] = X_test["T3"].fillna(t3_median)
X_test["TT4"] = X_test["TT4"].fillna(tt4_median)
X_test["T4U"] = X_test["T4U"].fillna(t4u_median)
X_test["FTI"] = X_test["FTI"].fillna(fti_median)

X_train.loc[ X_train["T4U_FTI_isnull"] ,"FTI"] = -999.0
X_test.loc[ X_test["T4U_FTI_isnull"] ,"FTI"] = -999.0

X_train.loc[ X_train["T4U_FTI_isnull"] ,"T4U"] = -999.0
X_test.loc[ X_test["T4U_FTI_isnull"] ,"T4U"] = -999.0


X_train.loc[ X_train["T4U_FTI_TSH_isnull"] ,"TSH"] = -999.0
X_test.loc[ X_test["T4U_FTI_TSH_isnull"] ,"TSH"] = -999.0



#----------------

# print(X_train.head())
# print(X_train.isnull().sum())

#---------------------

model = LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=5,
    subsample=0.80,
    colsample_bytree=0.80,
    objective = "binary",
    eval_metric="auc",
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

X_train["referral_source"] = X_train["referral_source"].astype("category")
X_test["referral_source"] = X_test["referral_source"].astype("category")




# print( X_train[numeric_cols].describe().T )
# print(len(X_train) , len(X_test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
oof_probas = np.zeros(len(X_train))

for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model.fit(X_tr, y_tr)
    oof_preds[val_idx] = model.predict(X_val)
    oof_probas[val_idx] = model.predict_proba(X_val)[:,1]



print("OOF Metrics on train (CV):")
print({
    "accuracy": accuracy_score(y_train, oof_preds),
    "precision": precision_score(y_train, oof_preds),
    "recall": recall_score(y_train, oof_preds),
    "f1": f1_score(y_train, oof_preds),
    "roc_auc": roc_auc_score(y_train, oof_probas),
    "pr_auc": average_precision_score(y_train, oof_probas)
})

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:,1]

print("Holdout Metrics on test:")
print({
    "accuracy": accuracy_score(y_test, y_test_pred),
    "precision": precision_score(y_test, y_test_pred),
    "recall": recall_score(y_test, y_test_pred),
    "f1": f1_score(y_test, y_test_pred),
    "roc_auc": roc_auc_score(y_test, y_test_proba),
    "pr_auc": average_precision_score(y_test, y_test_proba)
})

feature_importances_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

# Print or display the results
# print(feature_importances_df)


# print(dataset.head())

# together_missing_cols = ["FTI","T4U","TSH"]
# dataset["FTI_T4U_missing"] = dataset[together_missing_cols].isnull().all(axis=1).astype(int)
# print(dataset.groupby("FTI_T4U_missing")["Class"].mean())

#---------------------
# stage 2 , classify hypothyroid type

# print(dataset_hypo.head())
# print(dataset_hypo.isnull().sum())

dataset_hypo = dataset_hypo[ dataset_hypo["Class"] != "secondary_hypothyroid" ]

dataset_hypo["Class"] = dataset_hypo["Class"].map({"primary_hypothyroid":0 , "compensated_hypothyroid":1})

#dataset_hypo = dataset_hypo.reset_index(drop=True)


X_hypo = dataset_hypo.drop(columns=["Class"])
y_hypo = dataset_hypo["Class"]

# X_hypo["sex"] = X_hypo["sex"].fillna( X_hypo["sex"].mode()[0] )
# X_hypo["T4U"] = X_hypo["T4U"].fillna( X_hypo["T4U"].median() )
# X_hypo["T3"] = X_hypo["T3"].fillna( X_hypo["T3"].median() )
# X_hypo["FTI"] = X_hypo["FTI"].fillna( X_hypo["FTI"].median() )
# X_hypo["TT4"] = X_hypo["TT4"].fillna( X_hypo["TT4"].median())


# from sklearn.preprocessing import StandardScaler , OneHotEncoder

# pipeline = make_column_transformer(
#     (
#         StandardScaler(),
#         numeric_cols
#     ),
#     (
#         OneHotEncoder(sparse_output=False),
#         ohe_cols
#     )
# )

# X_hypo_scaled = pipeline.fit_transform(X_hypo)


# from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(X_hypo_scaled)

# print( pd.crosstab(clusters, y_hypo) )

#------------------------------------

X_train_hypo , X_test_hypo , y_train_hypo , y_test_hypo = train_test_split(
    X_hypo,
    y_hypo,
    test_size=0.2,
    stratify=y_hypo,
    random_state=42
)

#y_train_hypo = pd.Series(  np.random.permutation(y_train_hypo) ) # test for data leakage

sex_hypo_mode = X_train_hypo["sex"].mode()[0]
t4u_hypo_median = X_train_hypo["T4U"].median()
t3_hypo_median = X_train_hypo["T3"].median()
fti_hypo_median = X_train_hypo["FTI"].median()
tt4_hypo_median = X_train_hypo["TT4"].median()

X_train_hypo["sex"] = X_train_hypo["sex"].fillna(sex_hypo_mode)
X_train_hypo["T4U"] = X_train_hypo["T4U"].fillna(t4u_hypo_median)
X_train_hypo["T3"] = X_train_hypo["T3"].fillna(t3_hypo_median)
X_train_hypo["FTI"] = X_train_hypo["FTI"].fillna(fti_hypo_median)
X_train_hypo["TT4"] = X_train_hypo["TT4"].fillna(tt4_hypo_median)

X_test_hypo["sex"] = X_test_hypo["sex"].fillna(sex_hypo_mode)
X_test_hypo["T4U"] = X_test_hypo["T4U"].fillna(t4u_hypo_median)
X_test_hypo["T3"] = X_test_hypo["T3"].fillna(t3_hypo_median)
X_test_hypo["FTI"] = X_test_hypo["FTI"].fillna(fti_hypo_median)
X_test_hypo["TT4"] = X_test_hypo["TT4"].fillna(tt4_hypo_median)

X_train_hypo["referral_source"] = X_train_hypo["referral_source"].astype("category")
X_test_hypo["referral_source"] = X_test_hypo["referral_source"].astype("category")

# 

model_hypo = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=5,
    subsample=0.80,
    colsample_bytree=0.80,
    objective = "binary",
    eval_metric="auc",
    random_state=42,
    verbose=-1,
    n_jobs=-1
)


skf_hypo = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_preds_hypo = np.zeros(len(X_train_hypo))
oof_probas_hypo = np.zeros(len(X_train_hypo))

for train_idx, val_idx in skf_hypo.split(X_train_hypo, y_train_hypo):
    X_tr, X_val = X_train_hypo.iloc[train_idx], X_train_hypo.iloc[val_idx]
    y_tr, y_val = y_train_hypo.iloc[train_idx], y_train_hypo.iloc[val_idx]

    model_hypo.fit(X_tr, y_tr)
    oof_preds_hypo[val_idx] = model_hypo.predict(X_val)
    oof_probas_hypo[val_idx] = model_hypo.predict_proba(X_val)[:,1]


print("OOF Metrics on train (CV) hypo:")
print({
    "accuracy": accuracy_score(y_train_hypo, oof_preds_hypo),
    "precision": precision_score(y_train_hypo, oof_preds_hypo),
    "recall": recall_score(y_train_hypo, oof_preds_hypo),
    "f1": f1_score(y_train_hypo, oof_preds_hypo),
    "roc_auc": roc_auc_score(y_train_hypo, oof_probas_hypo),
    "pr_auc": average_precision_score(y_train_hypo, oof_probas_hypo)
})


model_hypo.fit(X_train_hypo, y_train_hypo)
y_test_hypo_pred = model_hypo.predict(X_test_hypo)
y_test_hypo_proba = model_hypo.predict_proba(X_test_hypo)[:,1]

print("Holdout Metrics on test (hypo):")
print({
    "accuracy": accuracy_score(y_test_hypo, y_test_hypo_pred),
    "precision": precision_score(y_test_hypo, y_test_hypo_pred),
    "recall": recall_score(y_test_hypo, y_test_hypo_pred),
    "specificity": recall_score(y_test_hypo, y_test_hypo_pred , pos_label=0), # specificity = recall_score(y_true, y_pred, pos_label=0)
    "f1": f1_score(y_test_hypo, y_test_hypo_pred),
    "roc_auc": roc_auc_score(y_test_hypo, y_test_hypo_proba),
    "pr_auc": average_precision_score(y_test_hypo, y_test_hypo_proba)
})

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test_hypo, y_test_hypo_pred).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
npv = tn / (tn + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"NPV: {npv:.3f}")
print(f"F1: {f1:.3f}")