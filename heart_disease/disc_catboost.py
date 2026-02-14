from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id", axis=1)
y = X.pop("Heart Disease")
y = y.map({"Presence": 1, "Absence": 0})

numeric_cols = [
    "ST depression",
    "Age",  "Cholesterol",
    "Max HR", "BP"
]

categorical_cols = [
    "EKG results","Thallium", "Chest pain type","Slope of ST","Number of vessels fluro"
]

binary_cols = [
    "Exercise angina","Sex", "FBS over 120" 
]


def get_bins(X,y,col):
    tree = DecisionTreeClassifier(max_leaf_nodes=5, min_samples_leaf=50) 
    tree.fit(X[[col]], y) 
    thresholds = tree.tree_.threshold 

    splits = sorted(thresholds[thresholds != -2])
    bins = [-np.inf] + splits + [np.inf]
    #print(col , " : " ,bins)
    return bins

X_disc = pd.DataFrame()

for ix,col in enumerate(numeric_cols):
    bins = get_bins(X,y,col)
    labels = [f"f_{ix}_bin_{i}" for i in range(len(bins)-1)]
    X_disc[ col + "_disc"] = pd.cut(X[col], bins=bins, labels=labels)

X_total = pd.concat( [X_disc , X[categorical_cols]] , axis=1 )

X_total = pd.concat( [X_total , X[binary_cols]] , axis=1 )

print(X_total.head())


params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 1500,
    "learning_rate": 0.07,
    "depth": 3,
    "l2_leaf_reg": 6,
    "subsample": 0.8,
    "random_strength": 1.0,
    "bootstrap_type": "Bernoulli",

    # CatBoost'a özgü kritik ayarlar
    "boosting_type": "Ordered",
    "one_hot_max_size": 10,   # binary feature’ları one-hot yapsın
    "max_ctr_complexity": 4,

    # stabilite
    "random_seed": 42,
    "verbose": 100
}

model = CatBoostClassifier(**params , cat_features=list(range(X_total.shape[1]))   )



skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

scores = cross_val_score(
    model,
    X_total,
    y,
    cv=skf,
    scoring="roc_auc",
    n_jobs=-1
    )
print(scores.mean())