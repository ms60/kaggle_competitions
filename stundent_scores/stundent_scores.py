import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline

from lightgbm import LGBMRegressor , LGBMClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV ,  train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder , StandardScaler , MinMaxScaler , FunctionTransformer

from sklearn.metrics import mean_absolute_error, r2_score , root_mean_squared_error

from xgboost import XGBRegressor


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")



print(train.head())
print(train.shape)
print(train.dtypes)

print(train.isnull().sum()) # no missing column ez

## feature engineering

train["study_efficiency"] = train["study_hours"] * (train["class_attendance"] / 100.0)
test["study_efficiency"] = test["study_hours"] * (test["class_attendance"] / 100.0)

train["sleep_efficiency"] = train["sleep_hours"] * train["sleep_quality"].map({"poor":1,"average":2,"good":3})
test["sleep_efficiency"] = test["sleep_hours"] * test["sleep_quality"].map({"poor":1,"average":2,"good":3})

train["student_discipline_score"] = 0.4 * train["study_hours"] + 0.3 * train["class_attendance"] +0.3 * train["sleep_efficiency"]
test["student_discipline_score"] = 0.4 * test["study_hours"] + 0.3 * test["class_attendance"] +0.3 * test["sleep_efficiency"]


train["facility_study_interaction"] = train["study_hours"] * train["facility_rating"].map({"low":1,"medium":2,"high":3})
test["facility_study_interaction"] = test["study_hours"] * test["facility_rating"].map({"low":1,"medium":2,"high":3})

train["low_attendance_flag"] = train["class_attendance"] < 75.0
test["low_attendance_flag"] = test["class_attendance"] < 75.0

train["sleep_deprivation_flag"] = (train["sleep_hours"] < 6 ) & ( train["study_hours"] > 5 )
test["sleep_deprivation_flag"] = (test["sleep_hours"] < 6 ) & ( test["study_hours"] > 5 )

train["over_study_flag"] = train["study_hours"] > 8
test["over_study_flag"] = test["study_hours"] > 8

train["study_hours_squared"] = train["study_hours"] * train["study_hours"]
test["study_hours_squared"] = test["study_hours"] * test["study_hours"]

train["sleep_hours_squared"] = train["sleep_hours"] * train["sleep_hours"]
test["sleep_hours_squared"] = test["sleep_hours"] * test["sleep_hours"]

# study_hours_squared
# sleep_hours_squared


###

cat_ordinal_cols = ["sleep_quality","facility_rating","exam_difficulty"]
cat_nominal_cols = ["gender","course","age","internet_access","study_method","low_attendance_flag","sleep_deprivation_flag","over_study_flag"]
num_cols = ["study_hours","class_attendance","sleep_hours","study_efficiency","sleep_efficiency","student_discipline_score","facility_study_interaction","study_hours_squared","sleep_hours_squared"]

target_col = ["exam_score"]

for col in cat_ordinal_cols:
    print(train[col].value_counts().index  )


print(train[num_cols+["exam_score"]].describe().T)

print("="*80)

for col in cat_nominal_cols:
    print( train[col].value_counts() / train.shape[0] )

#class imbalances : internet_access , course
print("="*80)
for col in cat_ordinal_cols:
    print( train[col].value_counts() / train.shape[0] )

#class imbalances: exam_difficulty


preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , cat_nominal_cols),
    (OrdinalEncoder(categories=[
        ["poor","average","good"],
        ["low","medium","high"],
        ["easy","moderate","hard"],
        ]),cat_ordinal_cols),
    (StandardScaler(),num_cols),
    remainder="drop"

)



# model = ElasticNet(
#     alpha=0.0005,
#     l1_ratio=0.9,
#     random_state=42
# )

model = LGBMRegressor(
    n_estimators=2500,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=6,
    feature_fraction=0.65,
    n_jobs=-1,
    verbosity=0,
    random_state=42
)

# model = XGBRegressor(
#     objective="reg:squarederror",  # regression için
#     n_jobs=-1,
#     random_state=42,
#     verbosity=0
# )

# #random sampling
# X_sample, _, y_sample, _ = train_test_split(
#     train.drop(["id","exam_score"] , axis =1) , train["exam_score"], train_size=0.2, random_state=42
# )


X_train , X_test  , y_train , y_test = train_test_split(train.drop(["id","exam_score"] , axis =1) , train["exam_score"] , test_size=0.075, random_state=42)

# data = preprocessor.fit_transform(X_train)
# print(data)

# model.fit(data,y_train)

# preds = model.predict(X_test)


# print("MAE:", mean_absolute_error(y_test, preds))
# print("R2 :", r2_score(y_test, preds))



pipe = make_pipeline(
    preprocessor,
    model
)

param_grid = {
    "lgbmregressor__n_estimators": [150, 250,500],
    "lgbmregressor__learning_rate": [0.01 , 0.25 , 0.05 , 0.1],
    "lgbmregressor__max_depth": [3, 6,9 , 15],
    "lgbmregressor__subsample": [0.8, 1.0],
    "lgbmregressor__colsample_bytree": [0.8, 1.0]
}

rs = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_grid,
    n_iter=5,          # 🔥 ASIL ZAMAN KONTROLÜ
    cv=3,              # 🔥 5 yerine 3
    random_state=42,
    n_jobs=-1,
    verbose=2
)


#rs.fit(X_train, y_train)
pipe.fit(X_train,y_train)



#best_pipe = rs.best_estimator_
#preds = best_pipe.predict(X_test)
preds = pipe.predict(X_test)



print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE",root_mean_squared_error(y_test,preds))
print("R2 :", r2_score(y_test, preds))
######

#pred_test = best_pipe.predict( test.drop("id",axis=1) )
pred_test = pipe.predict( test.drop("id",axis=1) )

result  = pd.DataFrame({"id":test["id"].to_list() , "exam_score":pred_test})
result.to_csv("result.csv",index=False)