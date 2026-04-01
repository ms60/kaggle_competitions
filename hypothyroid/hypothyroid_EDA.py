import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import arff

pd.set_option('display.max_columns', None)

data, meta = arff.loadarff("./data/dataset_57_hypothyroid.arff")

train = pd.DataFrame(data)

# byte string → normal string dönüşümü (gerekirse)
for col in train.select_dtypes([object]):
    train[col] = train[col].str.decode("utf-8")

print(train.head())
print(train.shape)

print(train["Class"].value_counts() )

print(train.isnull().sum())

print(train.dtypes)

print("-"*80)

#-------------------

binary_cols = ["on_thyroxine","query_on_thyroxine","on_antithyroid_medication","sick","pregnant","thyroid_surgery","I131_treatment","query_hypothyroid","query_hyperthyroid","lithium","goitre","tumor","hypopituitary","psych","TSH_measured","FTI_measured","TBG_measured"]
numeric_cols = ["age","TSH","T3","TT4","T4U","FTI","TBG"]
ohe_cols = ["referral_source"]



for col in binary_cols:
   train[col] = train[col].map({"f": 0, "t": 1})

train["sex"] = train["sex"].map({"F": 0, "M": 1})

#--------------------
# target analysis
sns.countplot(data=train["Class"]  )
plt.show()

# target is multiclass - 4
# target is unbalanced 

#--------------------
# univariate analysis

# fig , axes = plt.subplots(nrows= 2 , ncols= 4 , figsize=(10 , 6))
# axes = axes.flatten()

for col in binary_cols:
    print(train[col].value_counts())

#--------------------
fig , axes = plt.subplots(nrows= 2 , ncols= 4 , figsize=(10 , 6))
axes = axes.flatten()

for indx , col in enumerate(binary_cols[:8]):
    ax = axes[indx]
    sns.countplot(
        data=train,
        x=col,
        hue="Class",
        ax=ax
    )

fig.tight_layout(pad=1.0)
plt.xticks(rotation=45)
plt.show()


fig , axes = plt.subplots(nrows= 2 , ncols= 4 , figsize=(10 , 6))
axes = axes.flatten()

for indx , col in enumerate(binary_cols[8:16]):
    ax = axes[indx]
    sns.countplot(
        data=train,
        x=col,
        hue="Class",
        ax=ax
    )

fig.tight_layout(pad=1.0)
plt.xticks(rotation=45)
plt.show()

#----------------

print( train["TSH","T3","TT4","T4U","FTI"].isnull().sum() )

sns.heatmap(train)
plt.show()
