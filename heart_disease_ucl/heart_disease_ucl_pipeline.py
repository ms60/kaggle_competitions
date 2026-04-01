import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

preprocessor = make_column_transformer()
pipe = make_pipeline()

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

cleveland = pd.read_csv('data/heart+disease/processed.cleveland.data', 
                         header=None, names=columns)
hungarian = pd.read_csv('data/heart+disease/processed.hungarian.data', 
                         header=None, names=columns)
switzerland = pd.read_csv('data/heart+disease/processed.switzerland.data', 
                           header=None, names=columns)
va = pd.read_csv('data/heart+disease/processed.va.data', 
                  header=None, names=columns)

dataset = pd.concat([cleveland, hungarian, switzerland, va], ignore_index=True)

print(dataset.head())

dataset.replace('?', np.nan, inplace=True)

for col in dataset.columns:
    dataset[col] = dataset[col].astype(float)

X = dataset.drop('target', axis=1)
y = dataset['target'].apply(lambda x: 1 if x > 0 else 0).astype(int)

print(X.head())
print(y.head())

print(X.isnull().sum())

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 , stratify=y)
#----------------------
X_train_with_target = X_train.copy()
X_train_with_target['target'] = y_train

# EDA
# 1. target analysis
print(y_train.value_counts())
print(X_train.describe().T)
print(X_train.isnull().sum())
# sns.countplot(x=y )
# plt.show()
# 2. univariate analysis (only feature)
fig , axes = plt.subplots(nrows= 4 , ncols= 4 , figsize = (10,8))
axes = axes.flatten()

for idx,col in enumerate(X_train_with_target.columns):
    ax = axes[idx]
    sns.histplot(data = X_train_with_target, x = col , hue ="target" , kde=True , ax = ax , bins=15 )

fig.tight_layout(pad=1.0)
plt.xticks(rotation=45)
plt.show()
# 3. bivariate analysis (feature-feature)
# 4. 

# missing values analysis
# check if any pattern in missing data
 
X_train_missing = X_train.isnull()