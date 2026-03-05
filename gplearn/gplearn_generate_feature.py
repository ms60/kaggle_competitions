import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# 1️⃣ Veri üretelim
np.random.seed(42)

n = 1000
X = pd.DataFrame({
    "x1": np.random.uniform(-2, 2, n),
    "x2": np.random.uniform(-2, 2, n)
})

y = X["x1"]**2 + np.sin(X["x2"]) + np.random.normal(0, 0.1, n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2️⃣ SymbolicTransformer ile yeni feature üret
gp = SymbolicTransformer(
    generations=20,
    population_size=1000,
    hall_of_fame=10,
    n_components=5,
    function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos'),
    parsimony_coefficient=0.01,
    max_samples=0.9,
    random_state=42,
    verbose=1
)

gp.fit(X_train, y_train)

gp_features_train = gp.transform(X_train)
gp_features_test = gp.transform(X_test)

gp_feature_names = [f"gp_{i}" for i in range(gp_features_train.shape[1])]

gp_train_df = pd.DataFrame(gp_features_train, columns=gp_feature_names)
gp_test_df = pd.DataFrame(gp_features_test, columns=gp_feature_names)


print(gp_train_df)


# 🔹 Baseline
model_base = LGBMRegressor(random_state=42)
model_base.fit(X_train, y_train)

pred_base = model_base.predict(X_test)
rmse_base = np.sqrt(mean_squared_error(y_test, pred_base))

print("Baseline RMSE:", rmse_base)

X_train_aug = pd.concat([X_train.reset_index(drop=True),
                         gp_train_df.reset_index(drop=True)], axis=1)

X_test_aug = pd.concat([X_test.reset_index(drop=True),
                        gp_test_df.reset_index(drop=True)], axis=1)

model_aug = LGBMRegressor(random_state=42)
model_aug.fit(X_train_aug, y_train)

pred_aug = model_aug.predict(X_test_aug)
rmse_aug = np.sqrt(mean_squared_error(y_test, pred_aug))

print("Augmented RMSE:", rmse_aug)