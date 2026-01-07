import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer , KNNImputer , IterativeImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.compose import make_column_transformer , ColumnTransformer

from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder , StandardScaler , MinMaxScaler , FunctionTransformer
from sklearn.pipeline import make_pipeline

from lightgbm import LGBMRegressor , LGBMClassifier

from sklearn.metrics import mean_absolute_error, r2_score , accuracy_score , classification_report ,  precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


import re
def seperate_cabin(text):
    match = re.match(r"([a-z]+)([0-9]+)", text, re.I)
    if match:
        items = match.groups()
    return items

def handle_age(X):
    X = pd.Series(X[:,0].squeeze())
    bins = [0, 4, 8, 14, 18, 45, 60, np.inf]
    groups = [
        'AGE_0_4', 'AGE_4_8', 'AGE_8_14',
        'AGE_14_18', 'AGE_18_45', 'AGE_45_60', 'AGE_60_PLUS'
    ]
    return pd.DataFrame(  pd.cut(X, bins=bins, labels=groups , include_lowest=True) , columns=["AGE_GROUP"] )

"""
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

"""

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")


print(train.head())

print(train.isnull().sum())
print(train.shape)

######

print(train["Ticket"].sort_values())
print(train["Ticket"].nunique())

print(train.describe().T)


##################
#Feature Engineering

##Ticket
print( train["Ticket"].str.split(" ").apply(lambda row: len(row)).max() ) # max 3 
print(train.apply(lambda col : col["Ticket"].split(" ") , axis = 1 ) )
train["Ticket_F"] = train.apply(lambda col : col["Ticket"].split(" ")[0]  , axis = 1 )
train["Ticket_S"] = train.apply(lambda col : col["Ticket"].split(" ")[1] if len(col["Ticket"].split(" "))>1 else None , axis = 1 )
train["Ticket_T"] = train.apply(lambda col : col["Ticket"].split(" ")[2] if len(col["Ticket"].split(" "))>2 else None , axis = 1 )

train["Ticket_S"] = train.apply(lambda col: col["Ticket_F"] if col["Ticket_S"] is  None else col["Ticket_S"],axis = 1 )
train["Ticket_F"] = train.apply(lambda col: None if col["Ticket_F"]==col["Ticket_S"] else col["Ticket_F"] ,axis = 1 )



##Ticket

test["Ticket_F"] = test.apply(lambda col : col["Ticket"].split(" ")[0]  , axis = 1 )
test["Ticket_S"] = test.apply(lambda col : col["Ticket"].split(" ")[1] if len(col["Ticket"].split(" "))>1 else None , axis = 1 )
test["Ticket_T"] = test.apply(lambda col : col["Ticket"].split(" ")[2] if len(col["Ticket"].split(" "))>2 else None , axis = 1 )

test["Ticket_S"] = test.apply(lambda col: col["Ticket_F"] if col["Ticket_S"] is  None else col["Ticket_S"],axis = 1 )
test["Ticket_F"] = test.apply(lambda col: None if col["Ticket_F"]==col["Ticket_S"] else col["Ticket_F"] ,axis = 1 )


##Age


train = train[ train["Embarked"].notnull() ]


#print(train["Cabin"].apply(lambda row: seperate_cabin(row ) ))

# print(train["Cabin"].value_counts())

# print(train.head())

# print( train["Age"].squeeze() )
#####################

X_train , X_test  , y_train , y_test = train_test_split(train.drop(["PassengerId","Survived"] , axis =1) , train["Survived"] , test_size=0.2, random_state=60)

cat_ordinal_cols = ["Pclass","SibSp","Parch"]
cat_nominal_cols = ["Sex","Embarked"]
num_cols = ["Fare"]


age_pipeline = make_pipeline(
    SimpleImputer( strategy="median", add_indicator=True),
    FunctionTransformer(handle_age,validate=False),
    OneHotEncoder(handle_unknown='ignore')
)




ct_train = make_column_transformer(
    (age_pipeline, ["Age"]),  
    (StandardScaler() , num_cols),
    (OneHotEncoder(handle_unknown='ignore'), cat_nominal_cols),
    remainder='drop'
)

param_dist = {
    "num_leaves": [31, 63],         # küçük ağaçlar
    "max_depth": [15, 20,50],           # derinlik sınırlı
    "learning_rate": [0.1,0.01],         # hızlı öğrenme
    "n_estimators": [50, 100,300],      # az ağaç
    "min_child_samples": [20, 50 , 300],  # küçük yaprakları engelle
    "subsample": [0.8, 1.0],        # satır örnekleme
    "colsample_bytree": [0.8, 1.0]  # feature örnekleme
}

model = LGBMClassifier(
    objective="binary",
    random_state=60,
    n_jobs=1,        # tüm CPU yerine 1 kullan → paralellik bazen RandomizedSearchCV ile sorun çıkarıyor
    verbose=-1       # console logları kapat
)

rs = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=30,         # 10 kombinasyon → 1-2 dk sürer
    scoring="roc_auc",
    cv=3,              # 2-fold → hız kazanır
    verbose=0,
    n_jobs=1,          # paralellik kapalı → loop riskini azaltır
    random_state=42
)



rs.fit(ct_train.fit_transform(X_train), y_train)


X_test_transformed = ct_train.transform(X_test) 
y_pred = rs.predict(X_test_transformed)  # sınıf tahmini
y_proba = rs.predict_proba(X_test_transformed)[:, 1]

print({
    "accuracy": accuracy_score(y_test, y_pred),
    "precision":precision_score(y_test, y_pred),
    "recall":recall_score(y_test, y_pred),
    "f1":f1_score(y_test, y_pred),
    "roc_auc":roc_auc_score(y_test, y_proba),
    "pr_auc":average_precision_score(y_test,y_proba)
})


X_result_transformed = ct_train.transform(test)
y_result = rs.predict(X_result_transformed)  

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": y_result
})

submission.to_csv("result.csv", index=False)


"""
{'accuracy': 0.8202247191011236, 'precision': 0.8235294117647058, 'recall': 0.7368421052631579, 'f1': 0.7777777777777778, 'roc_auc': 0.872549019607843, 'pr_auc': 0.8490599899610817}
"""