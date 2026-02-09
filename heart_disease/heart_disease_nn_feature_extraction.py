from lightgbm import LGBMClassifier
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis = 1)
y = X.pop("Heart Disease")
y = y.map({"Presence":1 , "Absence":0})

num_cols = [
    "Age", "BP", "Cholesterol", "Max HR", "ST depression"
]

bin_cols = [
    "Sex", "FBS over 120", "Exercise angina"
]

cat_cols = [
    "Chest pain type", "EKG results",
    "Slope of ST", "Number of vessels fluro", "Thallium"
]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", TargetEncoder(smoothing=5), cat_cols),
        ("bin", "passthrough", bin_cols)
    ]
)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_p = preprocess.fit_transform(X_train, y_train)
X_val_p   = preprocess.transform(X_val)

X_train_t = torch.tensor(X_train_p, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_val_t = torch.tensor(X_val_p, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

    
class EncoderWithHead(nn.Module):
    def __init__(self, input_dim, emb_dim=32):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, emb_dim)
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return logits, z

model = EncoderWithHead(input_dim=X_train_t.shape[1], emb_dim=32)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 200

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    logits, _ = model(X_train_t)
    loss = criterion(logits, y_train_t)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f}")

model.eval()
with torch.no_grad():
    Z_train = model.encoder(X_train_t).cpu().numpy()
    Z_val   = model.encoder(X_val_t).cpu().numpy()

print(Z_train.shape, Z_val.shape)
# (n_train, 32), (n_val, 32)

def objective(trial):

    params= {
    "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]),
    "objective": "binary",
    "metric": "auc",
    "n_estimators": trial.suggest_int("n_estimators", 5000, 7000) ,#6405,#trial.suggest_int("n_estimators", 100, 7000),
    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5,log=True),
    "num_leaves": trial.suggest_int("num_leaves", 10, 256),
    "max_depth": trial.suggest_int("max_depth", 3, 5),#,3,#trial.suggest_int("max_depth", 3, 8),
    "min_child_samples": trial.suggest_int("min_child_samples", 10, 300,log=True),
    "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
    "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    "subsample": trial.suggest_float("subsample", 0.1, 1.0),
    #"subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0 , log=True),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0,log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0,log=True),
    #"max_bin": trial.suggest_int("max_bin", 64, 512),
    "random_state": 42,
    "verbosity": -1,
    }

    # x_train_proc = preprocess.fit_transform(X_train)
    # X_valid_proc = preprocess.transform(X_valid)

    model = LGBMClassifier(**params)
    model.fit(Z_train,y_train)

    #y_preds = model.predict(X_valid_proc)
    y_proba = model.predict_proba(Z_val)[:, 1]

    score = roc_auc_score(y_val, y_proba)
    
    
    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best ROC AUC:", study.best_value)
print("Best params:", study.best_params)