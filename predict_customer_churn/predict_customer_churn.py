from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier , early_stopping
import optuna
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler , TargetEncoder
from xgboost import XGBClassifier
from pytabkit import RealMLP_TD_Classifier


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

# for col in numeric_features:
#     freq = train[col].value_counts(normalize=True)
#     train[col].map(freq).fillna(0).astype('float32')
#     test[col].map(freq).fillna(0).astype('float32')


# train["InternetQuality"] =   train["OnlineSecurity"] + train["OnlineBackup"] + train["DeviceProtection"] + train["TechSupport"] + train["StreamingTV"] + train["StreamingMovies"]
# test["InternetQuality"] =  test["OnlineSecurity"] + test["OnlineBackup"] + test["DeviceProtection"] + test["TechSupport"] + test["StreamingTV"] + test["StreamingMovies"] 

# train["Security"] = train["OnlineSecurity"] * train["DeviceProtection"] * train["TechSupport"]
# test["Security"] = test["OnlineSecurity"] * test["DeviceProtection"] * test["TechSupport"]

# train['Expected_TotalCharges'] = train['tenure'] * train['MonthlyCharges']
# test['Expected_TotalCharges'] = test['tenure'] * test['MonthlyCharges']

# train['Charges_Difference'] = train['TotalCharges'] - train['Expected_TotalCharges']
# test['Charges_Difference'] = test['TotalCharges'] - test['Expected_TotalCharges']

# train['Cost_Per_Month_Tenure'] = train['TotalCharges'] / (train['tenure'] + 1)
# test['Cost_Per_Month_Tenure'] = test['TotalCharges'] / (test['tenure'] + 1)

# services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
#             'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
# train['Total_Services'] = train[services].sum(axis=1)
# test['Total_Services'] = test[services].sum(axis=1)


# train['Avg_Cost_Per_Service'] = train['MonthlyCharges'] / (train['Total_Services'] + 1)
# test['Avg_Cost_Per_Service'] = test['MonthlyCharges'] / (test['Total_Services'] + 1)



# contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
# train['Contract_Months'] = train['Contract'].map(contract_map).astype(float).fillna(1.0)
# test['Contract_Months'] = test['Contract'].map(contract_map).astype(float).fillna(1.0)


# train['Tenure_Contract_Ratio'] = train['tenure'] / train['Contract_Months']
# test['Tenure_Contract_Ratio'] = test['tenure'] / test['Contract_Months']

# train["Is_Safe_Customer"] = train["OnlineSecurity"] * train["TechSupport"]
# test["Is_Safe_Customer"] = test["OnlineSecurity"] * test["TechSupport"]


# train['Is_Auto_Payment'] = train['PaymentMethod'].astype(str).str.contains('automatic', case=False).astype(int)
# test['Is_Auto_Payment'] = test['PaymentMethod'].astype(str).str.contains('automatic', case=False).astype(int)

# train["Is_Streaming_Fan"] = train["StreamingMovies"] * train["StreamingTV"]
# test["Is_Streaming_Fan"] = test["StreamingMovies"] * test["StreamingTV"]

# train['Monthly_per_Contract_Month'] = train['MonthlyCharges'] / train['Contract_Months']
# test['Monthly_per_Contract_Month'] = test['MonthlyCharges'] / test['Contract_Months']


# train["tenure_treshold"] = (train["tenure"] >40.0).astype(int)
# test["tenure_treshold"] = (test["tenure"] >40.0).astype(int)

# train["tenure_squared"] = train["tenure"] * train["tenure"]
# test["tenure_squared"] = test["tenure"] * test["tenure"]








train["Churn"] = train["Churn"].map({"No": 0, "Yes": 1})

X = train.drop("id",axis=1)
y = X.pop("Churn")

X_test = test.drop("id",axis=1)



#numeric_features = [ "tenure_squared" ,"InternetQuality","Total_Services" ,"tenure", "MonthlyCharges", "TotalCharges"  , "Expected_TotalCharges" , "Charges_Difference" , "Cost_Per_Month_Tenure"  , "Avg_Cost_Per_Service","Contract_Months","Tenure_Contract_Ratio","Monthly_per_Contract_Month" ]
numeric_features = ["MonthlyCharges","TotalCharges"]
ohe_features = ["Contract","PaymentMethod"]
#log1p_features = ["MonthlyCharges", "TotalCharges","Expected_TotalCharges","Cost_Per_Month_Tenure","Avg_Cost_Per_Service","Tenure_Contract_Ratio","Monthly_per_Contract_Month"]

#targetEnc_features = ["InternetQuality","Security","Total_Services","Partner","Dependents" , "SeniorCitizen" , "PaperlessBilling"]

# te_pipeline = make_pipeline(
#     TargetEncoder(smooth="auto" , cv=5),
#     StandardScaler()
# )


# for col in log1p_features:
#     X[col] = np.log1p(X[col])
#     X_test[col] = np.log1p(X_test[col])



preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(), ohe_features),
    #( te_pipeline  , targetEnc_features),
    remainder="passthrough"
)

#baseline
# print(train.shape)
# print(train.isnull().sum())

# print(train[train.columns[:10]].head())
# print(train[train.columns[10:]].head())

# print( train["PhoneService"].value_counts() )

for col in ohe_features:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")






# X_train , X_valid , y_train , y_valid = train_test_split(X , y , test_size=0.075 , random_state=42 , stratify=y)
# X_train_2 , X_valid_2 , y_train_2 , y_valid_2 = train_test_split(X , y , test_size=0.1 , random_state=60 , stratify=y)

# X_train_proc = preprocessor.fit_transform(X_train , y_train)
# X_valid_proc = preprocessor.transform(X_valid)

# X_train_2_proc = preprocessor.fit_transform(X_train_2 , y_train_2)
# X_valid_2_proc = preprocessor.transform(X_valid_2)


skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def objective(trial):
    # params= {
    # "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]),
    # "objective": "binary",
    # "metric": "auc",
    # "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
    # "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5,log=True),
    # "num_leaves": trial.suggest_int("num_leaves", 10, 256),
    # "max_depth": trial.suggest_int("max_depth", 3, 8),#,3,#trial.suggest_int("max_depth", 3, 8),
    # "min_child_samples": trial.suggest_int("min_child_samples", 10, 300,log=True),
    # "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
    # "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    # "subsample": trial.suggest_float("subsample", 0.1, 1.0),
    # #"subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
    # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0 , log=True),
    # "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0,log=True),
    # "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0,log=True),
    # #"max_bin": trial.suggest_int("max_bin", 64, 512),
    # "random_state": 42,
    # "verbosity": -1,
    # }

    xgb_params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 5000),   
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5,log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0 , log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0,log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0,log=True),
        #"min_child_samples": trial.suggest_int("min_child_samples", 10, 300,log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        #"min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "max_bin": trial.suggest_int("max_bin", 64, 512),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_jobs': -1,
        'random_state': 42,
        'early_stopping_rounds': 200,
        'device': 'cuda',
        
        
        'enable_categorical': True, 
    }



    #model = LGBMClassifier(**params)
    model = XGBClassifier(**xgb_params)
    #model.fit(X_train,y_train)
    model.fit(X_train_proc,y_train,
              eval_set=[(X_valid_proc, y_valid)],
              verbose=0
        )
    

    #y_proba = model.predict_proba(X_valid)[:, 1]
    y_proba = model.predict_proba(X_valid_proc)[:, 1]

    score = roc_auc_score(y_valid, y_proba)

    #model2 = LGBMClassifier(**params)
    model2 = XGBClassifier(**xgb_params)
    #model2.fit(X_train_2,y_train_2)
    model2.fit(X_train_2_proc,y_train_2,
               eval_set=[(X_valid_2_proc, y_valid_2)],
               verbose=0)
    #y_proba2 = model2.predict_proba(X_valid_2)[:, 1]
    y_proba2 = model2.predict_proba(X_valid_2_proc)[:, 1]
    score2 = roc_auc_score(y_valid_2, y_proba2)



    # scores = cross_val_score(
    #     model,
    #     X, y,
    #     cv=skf,
    #     scoring="roc_auc",
    #     n_jobs=-1
    # )
    
    
    return  (score + score2) / 2 #score #scores.mean() # score 

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=60)

# print("Best ROC AUC:", study.best_value)
# print("Best params:", study.best_params)

# best_params = study.best_params

# best_params = {'boosting_type': 'gbdt', 'n_estimators': 8306, 'learning_rate': 0.1438185236737415, 'num_leaves': 76, 'max_depth': 5, 'min_child_samples': 31, 'min_child_weight': 0.00935466909015086, 'min_split_gain': 0.9132745987768063, 'subsample': 0.8048956160745835, 'colsample_bytree': 0.4319270317522004, 'reg_alpha': 0.2316055447728619, 'reg_lambda': 3.878992620351458}
#best_params = {'boosting_type': 'gbdt', 'n_estimators': 3044, 'learning_rate': 0.08689656825820848, 'num_leaves': 210, 'max_depth': 4, 'min_child_samples': 18, 'min_child_weight': 0.011974187303327171, 'min_split_gain': 0.07422986122739264, 'subsample': 0.35954490266830175, 'colsample_bytree': 0.5024304816412929, 'reg_alpha': 0.034860128015873, 'reg_lambda': 0.05003006774124102}

# best_params = {'boosting_type': 'gbdt', 'n_estimators': 2175, 'learning_rate': 0.22087603717987528, 'num_leaves': 113, 'max_depth': 4, 'min_child_samples': 292, 'min_child_weight': 1.0123752022896289, 'min_split_gain': 0.0892272541091517, 'subsample': 0.5583147585364083, 'colsample_bytree': 0.3960494092561661, 'reg_alpha': 0.20120059760688364, 'reg_lambda': 2.9044688946621844}
#best_params = {'boosting_type': 'gbdt', 'n_estimators': 3207, 'learning_rate': 0.07931040793346719, 'num_leaves': 84, 'max_depth': 5, 'min_child_samples': 85, 'min_child_weight': 0.22755937045227428, 'min_split_gain': 0.061571078804342816, 'subsample': 0.7353569513590934, 'colsample_bytree': 0.5335276884646094, 'reg_alpha': 0.6154086150021465, 'reg_lambda': 0.3903499106248194}


#best_params = {'boosting_type': 'gbdt', 'n_estimators': 2505, 'learning_rate': 0.032916691149443934, 'num_leaves': 99, 'max_depth': 5, 'min_child_samples': 123, 'min_child_weight': 0.5048487526291471, 'min_split_gain': 0.006161204885943813, 'subsample': 0.4972410334806299, 'colsample_bytree': 0.441518570734105, 'reg_alpha': 0.28809811666299795, 'reg_lambda': 0.09981407573393641}

#{'boosting_type': 'gbdt', 'n_estimators': 3238, 'learning_rate': 0.03687901299540242, 'num_leaves': 135, 'max_depth': 4, 'min_child_samples': 68, 'min_child_weight': 0.10818451686535, 'min_split_gain': 0.02019299738819494, 'subsample': 0.46031029099561654, 'colsample_bytree': 0.4454739470307919, 'reg_alpha': 0.0010422307016903032, 'reg_lambda': 0.9563906689101879}
#{'boosting_type': 'gbdt', 'n_estimators': 3134, 'learning_rate': 0.05562272025378019, 'num_leaves': 125, 'max_depth': 5, 'min_child_samples': 86, 'min_child_weight': 0.003994115917666145, 'min_split_gain': 0.1536133239430104, 'subsample': 0.11288445138079785, 'colsample_bytree': 0.689517067015335, 'reg_alpha': 0.0012878909540679875, 'reg_lambda': 1.4385982271429836}
#best_params =  {'boosting_type': 'gbdt', 'n_estimators': 3686, 'learning_rate': 0.042233987505403366, 'num_leaves': 135, 'max_depth': 5, 'min_child_samples': 76, 'min_child_weight': 0.17465047574971193, 'min_split_gain': 0.030362576690836283, 'subsample': 0.46027777923464497, 'colsample_bytree': 0.4420788366501833, 'reg_alpha': 0.006730096397606108, 'reg_lambda': 1.0041760611014332}


#{'boosting_type': 'gbdt', 'n_estimators': 4302, 'learning_rate': 0.017193861222618758, 'num_leaves': 137, 'max_depth': 5, 'min_child_samples': 14, 'min_child_weight': 4.442546949278816, 'min_split_gain': 0.09647065329126837, 'subsample': 0.26423192124646844, 'colsample_bytree': 0.33704310570169144, 'reg_alpha': 0.006167337218904456, 'reg_lambda': 0.18354503392824864}
#best_params = {'boosting_type': 'gbdt', 'n_estimators': 1960, 'learning_rate': 0.07960117378294493, 'num_leaves': 139, 'max_depth': 4, 'min_child_samples': 75, 'min_child_weight': 0.7762796926158728, 'min_split_gain': 0.14812607431028918, 'subsample': 0.5453171083097746, 'colsample_bytree': 0.2906440355658162, 'reg_alpha': 0.012083576586514827, 'reg_lambda': 0.4550524224900979}
#best_params = {'boosting_type': 'gbdt', 'n_estimators': 4910, 'learning_rate': 0.01458978383427039, 'num_leaves': 45, 'max_depth': 5, 'min_child_samples': 11, 'min_child_weight': 4.697798759563624, 'min_split_gain': 0.08215945522822365, 'subsample': 0.1826636157326016, 'colsample_bytree': 0.3004733863428752, 'reg_alpha': 0.0027709022383171325, 'reg_lambda': 0.17550013459955266}


#best_params = {'boosting_type': 'gbdt', 'n_estimators': 4967, 'learning_rate': 0.00982101577044508, 'num_leaves': 171, 'max_depth': 6, 'min_child_samples': 21, 'min_child_weight': 7.6187438604004845, 'min_split_gain': 0.19697909076020012, 'subsample': 0.7307448230595652, 'colsample_bytree': 0.18611740562769047, 'reg_alpha': 0.012421016741057854, 'reg_lambda': 0.4164730346740021}


#{'boosting_type': 'gbdt', 'n_estimators': 2542, 'learning_rate': 0.027755393586866184, 'num_leaves': 63, 'max_depth': 8, 'min_child_samples': 43, 'min_child_weight': 0.0022906718407571755, 'min_split_gain': 0.13494349893099256, 'subsample': 0.8858865243497533, 'colsample_bytree': 0.10004256598152596, 'reg_alpha': 0.00857972565495584, 'reg_lambda': 1.682350814051624}
#{'boosting_type': 'gbdt', 'n_estimators': 4993, 'learning_rate': 0.03760831853812597, 'num_leaves': 252, 'max_depth': 7, 'min_child_samples': 52, 'min_child_weight': 0.004732465204721288, 'min_split_gain': 0.4705286179307953, 'subsample': 0.5125594014161537, 'colsample_bytree': 0.10131370769826743, 'reg_alpha': 0.006502285011920208, 'reg_lambda': 0.44560309871998627}
#best_params = {'boosting_type': 'gbdt', 'n_estimators': 3531, 'learning_rate': 0.019092789186995533, 'num_leaves': 65, 'max_depth': 8, 'min_child_samples': 23, 'min_child_weight': 0.00271024033162914, 'min_split_gain': 0.1271077856948274, 'subsample': 0.9023819406613406, 'colsample_bytree': 0.14469052704869442, 'reg_alpha': 0.04252751547906813, 'reg_lambda': 0.08874709690367247}


#{'boosting_type': 'gbdt', 'n_estimators': 3771, 'learning_rate': 0.029174802652806052, 'num_leaves': 241, 'max_depth': 5, 'min_child_samples': 55, 'min_child_weight': 0.061091559731823494, 'min_split_gain': 0.2719488536256223, 'subsample': 0.27386990891578866, 'colsample_bytree': 0.20512280387868312, 'reg_alpha': 0.023447658894673703, 'reg_lambda': 0.047880523747092466}
#{'boosting_type': 'gbdt', 'n_estimators': 3916, 'learning_rate': 0.02981023027714413, 'num_leaves': 222, 'max_depth': 6, 'min_child_samples': 92, 'min_child_weight': 0.06630506003998742, 'min_split_gain': 0.26195763479276846, 'subsample': 0.3745591227826521, 'colsample_bytree': 0.18441590816379316, 'reg_alpha': 2.6385122544835555, 'reg_lambda': 0.0031478473483066856}
#best_params = {'boosting_type': 'gbdt', 'n_estimators': 3401, 'learning_rate': 0.016909161626697644, 'num_leaves': 241, 'max_depth': 6, 'min_child_samples': 45, 'min_child_weight': 0.059106121633267915, 'min_split_gain': 0.2515646958558119, 'subsample': 0.21823556612043027, 'colsample_bytree': 0.18285452373017372, 'reg_alpha': 2.6611665087360015, 'reg_lambda': 0.002514575755093516}

best_params = {'boosting_type': 'gbdt', 'n_estimators':20000, 'learning_rate': 0.015,  'max_depth': 8,   'subsample': 0.8, 'colsample_bytree': 0.8 , 'max_bin':16000 }
best_params.update({
    "objective": "binary",
    "metric": "auc",
    "early_stopping_rounds":200,
    "random_state": 42,
    #'device': 'cuda',
    "verbosity": -1,

})

xgb_params = {
    'n_estimators': 60000,      
    'learning_rate': 0.01,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree':0.8,
    'max_bin':16000,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    #'early_stopping_rounds': 200,
    'device': 'cuda',
    
    'enable_categorical': True,
}

lgbm_params = {
    'n_estimators': 60000,
    'learning_rate': 0.01,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree':0.8,
    'max_bin':16000,
    'objective': 'binary',
    'metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    #'early_stopping_rounds': 200,
    #'device': 'cuda',
    'verbosity':-1,
}


# xgb_params = {'n_estimators': 4941, 'learning_rate': 0.024177187240319083, 'max_depth': 5, 'subsample': 0.9247050359150862, 'colsample_bytree': 0.2067854023851052, 'reg_alpha': 4.953183172415591, 'reg_lambda': 0.05426657094295068, 'min_child_weight': 0.0037006781291288346, 'max_bin': 362}
# xgb_params.update({
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',
#         'n_jobs': -1,
#         'random_state': 42,
#         'early_stopping_rounds': 200,
#         'device': 'cuda',
#         'enable_categorical': True, 
# })



# best_model = LGBMClassifier(**best_params)

# pipe = make_pipeline(
#     preprocessor,
#     best_model
# )



# scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
# print(scores)
# print(scores.mean())
# print(scores.std())


# pipe.fit(X,y)

# y_proba = pipe.predict_proba(X_test)[:, 1]

# result = pd.DataFrame({"id": test["id"], "Churn": y_proba})
# result.to_csv("submission.csv", index=False)


# X['f1'] = (X["tenure"] < 8.10).astype(int)
# X_test['f1'] = (X_test["tenure"] < 8.10).astype(int)

# xgb_params_best = {'n_estimators': 18965, 'learning_rate': 0.027755495257911617, 'max_depth': 3, 'subsample': 0.9908219823763396, 'colsample_bytree': 0.9448335609443502, 'reg_alpha': 1.6491249286990706, 'reg_lambda': 0.1357210672004791, 'min_child_weight': 0.027343037682091197, 'max_bin': 16920}

# xgb_params_best.update({
#     #"max_depth":3,
#     'objective': 'binary:logistic',
#     'eval_metric': 'auc',
#     'n_jobs': -1,
#     'random_state': 42,
#     'early_stopping_rounds': 200,
#     'device': 'cuda',
#     'enable_categorical': True,
# })

xgb_params_best = dict(
    n_estimators=100_000,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=200,
    device="cuda",
    #max_bin=16000,
)

real_mlp_params = {
    'device': 'cuda',
    'random_state': 42,
    'verbosity': 2,
    'n_epochs': 3,
    'batch_size': 256,
    'n_ens': 8,
    'val_metric_name': '1-auc_ovr',
    'use_early_stopping': True,
    'early_stopping_additive_patience': 20,
    'early_stopping_multiplicative_patience': 1,
    'act': "mish",
    'embedding_size': 6,
    'first_layer_lr_factor': 0.25,
    'hidden_sizes': "rectangular",
    'hidden_width': 352,
    'lr': 0.075,
    'ls_eps': 0.01,
    'ls_eps_sched': "coslog4",
    'max_one_hot_cat_size': 18,
    'n_hidden_layers': 4,
    'p_drop': 0.05,
    'p_drop_sched': "flat_cos",
    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,
    'scale_lr_factor': 2.24,
    'sq_mom': 0.988,
    'wd': 0.0236,
}
#----------------------

for col in X.columns:
    if col not in numeric_features:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")


#pseudo labeling
pseudo = pd.read_csv("./result.csv")["Churn"]

print( pseudo[(pseudo > 0.95) | (pseudo < 0.05) ] )

mask = (pseudo > 0.95) | (pseudo < 0.05)

pseudo_X = X_test[mask]
pseudo_y = (pseudo[mask] > 0.5).astype(int)

print(pseudo_X)
print(pseudo_y)

X_FE = pd.concat([X, pseudo_X])
y_FE = pd.concat([y, pseudo_y])


#---------------------
#generate oofs

oof_preds = np.zeros(len(X), dtype=float) 
test_preds = np.zeros(len(X_test), dtype=float) 

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)




for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

    print(f"Fold {fold+1}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # X_train_proc = preprocessor.fit_transform(X_train)
    # X_valid_proc = preprocessor.transform(X_valid)

    # X_test_proc = preprocessor.transform(X_test)



    # model = LGBMClassifier(**lgbm_params)
    # model.fit(
    #     X_train, y_train,
    #     eval_set=[(X_valid, y_valid)],
    #     eval_metric="auc",
    #     callbacks=[early_stopping(300, verbose=False)] 
    # )

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=0 
    )

    # model = RealMLP_TD_Classifier(**real_mlp_params)
    # model.fit(
    #     X_train, y_train,
    #     X_valid, y_valid
         
    # )

    # model = CatBoostClassifier(**CATBOOST_PARAMS)
    # model.fit(
    #     X_train, y_train,
    #     eval_set=[(X_valid, y_valid)],
    #     cat_features= [i for i in X.columns.tolist() if i not in numeric_features]  , 
    #     verbose=0 
    #     )



    # model = RealMLP_TD_Classifier(**REALMLP_TD_PARAMS)
    # model.fit(
    #     X_train, y_train,
    #     #eval_set=[(X_valid, y_valid)],
    #     #verbosity=-1 
    #     )

    oof_preds[valid_idx] = model.predict_proba(X_valid)[:,1]
    test_preds += model.predict_proba(X_test)[:,1] / skf.n_splits


oof_X = pd.DataFrame({"oof_pred": oof_preds})
oof_test = pd.DataFrame({"oof_pred_test": test_preds})

oof_X.to_csv("./stack/X_21_oof.csv",index=False)
oof_test.to_csv("./stack/X_21_oof_test.csv",index=False)