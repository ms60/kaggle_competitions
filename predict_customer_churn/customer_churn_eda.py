from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


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

#-----------------------------------------
train["Churn"] = train["Churn"].map({"No": 0, "Yes": 1})

X = train.drop("id",axis=1)
y = X.pop("Churn")

X_test = test.drop("id",axis=1)


#-----------------------------------------

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
ohe_features = ["Contract","PaymentMethod"]
binary_features = [col for col in X.columns if col not in numeric_features + ohe_features ]


#-----------------------------------------




# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

# for indx,feature in enumerate(ohe_features):
#     sns.countplot(
#         data=train,
#         x=feature,
#         hue="Churn",
#         ax=axes[indx]
#     )

# fig.tight_layout(pad=1.0)
# plt.xticks(rotation=45)
# plt.show()

#-----------------------------

# fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(18, 12))
# axes = axes.flatten()

# for indx,feature in enumerate(binary_features):
#     ax = axes[indx]

#     sns.countplot(
#         data=train,
#         x=feature,
#         hue="Churn",
#         ax=ax
#     )

    

#     # kategori toplamlarını hesapla
#     totals = train.groupby(feature)["Churn"].count()

#     for p in ax.patches:
#         height = p.get_height()
#         x = p.get_x() + p.get_width()/2

#         # hangi kategoriye ait olduğunu bul
#         category = int(round(x))
#         total = totals.iloc[category]

#         ratio = height / total

#         ax.annotate(
#             f"{ratio:.2f}",
#             (x, height),
#             ha='center',
#             va='bottom',
#             fontsize=9
#         )

# fig.tight_layout(pad=1.0)
# for ax in axes:
#     ax.tick_params(axis='x', rotation=45)
# plt.show()

#-----------------------------

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 12))
# axes = axes.flatten()


# for indx,feature in enumerate(numeric_features):
#     sns.histplot(
#         data=train,
#         x=feature,
#         hue="Churn",
#         bins=50,
#         kde=False,
#         stat="density",
#         common_norm=False,
#         ax=axes[indx]
    
#     )

# fig.tight_layout(pad=1.0)
# plt.xticks(rotation=45)
# plt.show()

#-------------------------------------------

print(X[binary_features].corr())

sns.heatmap(X[binary_features].corr())
plt.show()