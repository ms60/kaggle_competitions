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
from sklearn.metrics import roc_auc_score, accuracy_score

# ==========================
# 1. Load dataset
# ==========================
train = pd.read_csv("./data/train.csv")  
test = pd.read_csv("./data/test.csv")    


X = train.drop("id", axis=1)
y = X.pop("Heart Disease")


y = y.map({"Presence": 1, "Absence": 0})

# ==========================
# 2. Determine feature/column types
# ==========================
num_cols = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]  # numeric cols
bin_cols = ["Sex", "FBS over 120", "Exercise angina"]               # binary cols
cat_cols = [                                                        # categorical cols
    "Chest pain type", "EKG results",
    "Slope of ST", "Number of vessels fluro", "Thallium"
]

# ==========================
# 3. Preprocessing pipeline
# ==========================

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", TargetEncoder(smoothing=5), cat_cols),
        ("bin", "passthrough", bin_cols)
    ]
)

# ==========================
# 4. Train/Validation Split
# ==========================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


X_train_p = preprocess.fit_transform(X_train, y_train)
X_val_p = preprocess.transform(X_val)

# ==========================
# 5. Torch tensor dönüşümü
# ==========================
# Convert datas to pytorch tensor
X_train_t = torch.tensor(X_train_p, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)

X_val_t = torch.tensor(X_val_p, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

# ==========================
# 6. MLP Model Definition 
# ==========================
class HeartDiseaseMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # nn.Sequential:  puts the layers in order
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),   # initial -> convert input dimensions to 128-neurons layer
            nn.ReLU(),                   # activation: ReLU (handles nonlinearity)
            nn.BatchNorm1d(128),         # Batch Normalization (stabilize the train)
            nn.Dropout(0.3),             # %30 dropout: reduces overfitting

            nn.Linear(128, 64),          # 128 -> 64 neurons hidden layer
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 1)             # output layer: 1 neuron (binary classification)
        )

    def forward(self, x):
        return self.net(x)               # input → nn → output

# input_dim = feature sayısı

input_dimension = X_train_t.shape[1]
model = HeartDiseaseMLP(input_dimension)

# ==========================
# 7. Train setup
# ==========================
optimizer = Adam(model.parameters(), lr=1e-3)  # ADAM optimizer: momentum + adaptive learning rate
# lr=1e-3 -> learning rate. 0.001 typical initial value.

criterion = BCEWithLogitsLoss()
# BCEWithLogitsLoss = Binary Cross Entropy + Sigmoid 
# (model son katmanda sigmoid uygulamadığı için bu loss otomatik olarak içeriyor)

EPOCHS = 170  # model will be trained 170 times

# ==========================
# 8. Train loop
# ==========================
for epoch in range(EPOCHS):
    model.train()               # training mode
    optimizer.zero_grad()       # make zero previous gradients

    logits = model(X_train_t).squeeze()   # model output (raw logits)
    loss = criterion(logits, y_train_t)   # loss calculation (BCE with logits)
    loss.backward()                       # backprop (calculate gradient)
    optimizer.step()                      # update weights

    # ---------- Validation ----------
    model.eval()                          # eval mode 
    with torch.no_grad():                 # gradient hesaplama kapalı (hız için)
        val_logits = model(X_val_t).squeeze()
        val_probs = torch.sigmoid(val_logits)  # sigmoid -> 0–1 probabilities

        auc = roc_auc_score(y_val, val_probs.numpy())             # ROC-AUC metrics
        acc = accuracy_score(y_val, (val_probs > 0.5).numpy())    # accuracy with 0.5 threshold

    # print metrics every 5 epochs
    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | AUC={auc:.4f} | acc={acc:.4f}")

# # ==========================
# # 9. Test verisi hazırlama
# # ==========================
# # Aynı preprocess işlemini test verisine de uygula
# X_test_p = preprocess.transform(test.drop("id", axis=1))
# X_test_p = X_test_p.toarray() if hasattr(X_test_p, "toarray") else X_test_p
# X_test_t = torch.tensor(X_test_p, dtype=torch.float32)

# # ==========================
# # 10. Tahmin ve kayıt
# # ==========================
# model.eval()  # testte dropout / batchnorm kapalı
# with torch.no_grad():
#     test_logits = model(X_test_t).squeeze()
#     test_probs = torch.sigmoid(test_logits)  # olasılığa çevir

# # Sonuçları DataFrame'e yaz
# result = pd.DataFrame({
#     "id": test["id"],
#     "Heart Disease": test_probs.numpy()   # her birey için kalp hastalığı olasılığı
# })

# # CSV'ye kaydet
# result.to_csv("result_dl.csv", index=False)
# print("✅ result_dl.csv oluşturuldu.")
