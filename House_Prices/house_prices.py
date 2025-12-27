import pandas as pd
import numpy as np
import seaborn as sns

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print(train.head())