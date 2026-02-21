from PIL import Image
import numpy as np
import pandas as pd
import os

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



train_image_list = []
test_image_list = []

for dirname, _, filenames in os.walk('./data/train'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        train_image_list.append(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk('./data/test'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        test_image_list.append(os.path.join(dirname, filename))


train = pd.DataFrame( {"path": train_image_list } )
test = pd.DataFrame({"path":test_image_list})


#train
train["digit"] = train["path"].apply(lambda x: int(x.split("_")[1]) )
train["digit_id"] = train["path"].apply(lambda x: int(x.split("_")[2].split(".")[0] ) )
train["array"] = train["path"].apply(lambda x : np.array(Image.open(x))  )
train["array_flatten"] = train["array"].apply(lambda x : x.reshape(-1) )

train = train.sort_values(['digit', 'digit_id'])
train = train.reset_index(drop=True)

#test


test["array"] = test["path"].apply(lambda x : np.array(Image.open(x))  )
test["array_flatten"] = test["array"].apply(lambda x : x.reshape(-1) )




X = np.vstack(train["array_flatten"].values) / 255.0
y = train["digit"]

X_test = np.vstack(test["array_flatten"].values) / 255.0



X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)
print("Accuracy:", accuracy_score(y_valid, y_pred))

y_pred_test = model.predict(X_test)

result = pd.DataFrame( {"image_id": test["path"].str.split("/").str[-1] , "label": y_pred_test} )
result.to_csv("result.csv",index=False)