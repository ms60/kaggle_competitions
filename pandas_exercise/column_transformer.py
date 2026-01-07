import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.compose import make_column_transformer , ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


df = sns.load_dataset("titanic")[["fare","embarked","sex","age"]].tail()

ohe = OneHotEncoder()
imp = SimpleImputer()


print(df)

ct = make_column_transformer(
    ( ohe ,["embarked","sex"]),
    ( imp ,["age"]),
    remainder="passthrough"
)
# ordering is not accourding to df , its according to arguments of ct

print(ct.fit_transform(df))

# transformer = ColumnTransformer(
#     (),
#     (),
#     ()
# )