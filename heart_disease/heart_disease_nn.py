import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna

# ==========================
# 1. Veri yükleme ve hazırlık
# ==========================
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
X = train.drop("id", axis=1)
y = X.pop("Heart Disease").map({"Presence": 1, "Absence": 0})

num_cols = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
bin_cols = ["Sex", "FBS over 120", "Exercise angina"]
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

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_p = preprocess.fit_transform(X_train, y_train)
X_val_p = preprocess.transform(X_val)

X_train_p = X_train_p.toarray() if hasattr(X_train_p, "toarray") else X_train_p
X_val_p = X_val_p.toarray() if hasattr(X_val_p, "toarray") else X_val_p

X_train_t = torch.tensor(X_train_p, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
X_val_t = torch.tensor(X_val_p, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

# ==========================
# 2. Model tanımı (dinamik)
# ==========================
class HeartDiseaseMLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, dropout, use_bn):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden1),
            nn.ReLU()
        ]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden1))
        layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden1, hidden2))
        layers.append(nn.ReLU())
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden2))
        layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==========================
# 3. Objective fonksiyonu (Optuna için)
# ==========================
def objective(trial):
    # Optuna'nın deneyeceği parametreler
    hidden1 = trial.suggest_int("hidden1", 64, 256)
    hidden2 = trial.suggest_int("hidden2", 32, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    use_bn = trial.suggest_categorical("use_bn", [True, False])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = 100  # hızlı arama için 50 epoch sabit

    model = HeartDiseaseMLP(
        input_dim=X_train_t.shape[1],
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout,
        use_bn=use_bn
    )

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()

    # Eğitim
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t).squeeze()
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).squeeze()
        val_probs = torch.sigmoid(val_logits)
        auc = roc_auc_score(y_val, val_probs.numpy())

    return auc  # Optuna maximize edecek (yüksek AUC iyi)

# ==========================
# 4. Optuna çalışma
# ==========================
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=30, show_progress_bar=True)

# print("✅ En iyi deneme:")
# print(study.best_trial.params)
# print(f"Best AUC: {study.best_value:.4f}")

# # ==========================
# # 5. En iyi modeli yeniden eğit
# # ==========================
# best_params = study.best_trial.params

# best_model = HeartDiseaseMLP(
#     input_dim=X_train_t.shape[1],
#     hidden1=best_params["hidden1"],
#     hidden2=best_params["hidden2"],
#     dropout=best_params["dropout"],
#     use_bn=best_params["use_bn"]
# )

# optimizer = Adam(best_model.parameters(), lr=best_params["lr"])
# criterion = BCEWithLogitsLoss()

# for epoch in range(100):
#     best_model.train()
#     optimizer.zero_grad()
#     logits = best_model(X_train_t).squeeze()
#     loss = criterion(logits, y_train_t)
#     loss.backward()
#     optimizer.step()

# best_model.eval()
# with torch.no_grad():
#     val_logits = best_model(X_val_t).squeeze()
#     val_probs = torch.sigmoid(val_logits)
#     auc = roc_auc_score(y_val, val_probs.numpy())

# print(f"Final validation AUC: {auc:.4f}")

# ==========================
# 6. Test verisi ve Submission
# ==========================

best_params = {'hidden1': 190, 'hidden2': 92, 'dropout': 0.15709081220627946, 'use_bn': True, 'lr': 0.00895869821885461}

best_model = HeartDiseaseMLP(
    input_dim=X_train_t.shape[1],
    hidden1=best_params["hidden1"],
    hidden2=best_params["hidden2"],
    dropout=best_params["dropout"],
    use_bn=best_params["use_bn"]
)

optimizer = Adam(best_model.parameters(), lr=best_params["lr"])
criterion = BCEWithLogitsLoss()

EPOCHS = 250
for epoch in range(EPOCHS):
    best_model.train()
    optimizer.zero_grad()

    logits = best_model(X_train_t).squeeze()
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 25 == 0:
        best_model.eval()
        with torch.no_grad():
            val_logits = best_model(X_val_t).squeeze()
            val_probs = torch.sigmoid(val_logits)
            auc = roc_auc_score(y_val, val_probs.numpy())
        print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | AUC={auc:.4f}")

X_test_p = preprocess.transform(test.drop("id", axis=1))
X_test_p = X_test_p.toarray() if hasattr(X_test_p, "toarray") else X_test_p
X_test_t = torch.tensor(X_test_p, dtype=torch.float32)

best_model.eval()
with torch.no_grad():
    test_logits = best_model(X_test_t).squeeze()
    test_probs = torch.sigmoid(test_logits)

submission = pd.DataFrame({
    "id": test["id"],
    "Heart Disease": test_probs.numpy()
})
submission.to_csv("result_best_model.csv", index=False)
print("✅ result_best_model.csv oluşturuldu.")