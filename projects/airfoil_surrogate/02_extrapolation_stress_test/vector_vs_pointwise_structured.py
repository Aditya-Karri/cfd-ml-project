import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Architecture Search (Extrapolation Stress Test)
# Objective: Compare if predicting the entire pressure curve at once (Vector) 
# yields better extrapolation fidelity than predicting coordinates one by one (Pointwise)

# Setup
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "airfoil_dataset.npz")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("Loading aerodynamic dataset")
data = np.load(DATA_PATH)
X_raw = data["X"]
y_cp = data["y_cp"]
x_grid = data["x_grid"]

aoa = X_raw[:, 0].reshape(-1, 1)
print(f"Loaded {len(aoa)} cases. Target shape: {y_cp.shape}")

NUM_POINTS = y_cp.shape[1] // 2  
total_cp_points = y_cp.shape[1]  

# Structured split (Extrapolation)
aoa_flat = aoa.flatten()
train_mask = aoa_flat <= 8
val_mask   = (aoa_flat > 8) & (aoa_flat <= 12)
test_mask  = aoa_flat > 12

aoa_train, aoa_val, aoa_test = aoa[train_mask], aoa[val_mask], aoa[test_mask]
cp_train_flat, cp_val_flat, cp_test_flat = y_cp[train_mask], y_cp[val_mask], y_cp[test_mask]

# Scale inputs (fit only on training data)
input_scaler = StandardScaler()
aoa_train_scaled = input_scaler.fit_transform(aoa_train)
aoa_val_scaled = input_scaler.transform(aoa_val)
aoa_test_scaled = input_scaler.transform(aoa_test)

# Save scaler
joblib.dump(input_scaler, os.path.join(MODELS_DIR, "aoa_scaler_structured.pkl"))

early_stop = tf.keras.callbacks.EarlyStopping(
    patience=30, 
    restore_best_weights=True,
    monitor="val_loss"
)

# Architecture A: Vector output
print(f"\nTraining Model A: Predicting the full pressure distribution all at once ({total_cp_points} outputs)")
model_A = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(total_cp_points, activation="linear")
])

model_A.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

history_A = model_A.fit(
    aoa_train_scaled, cp_train_flat,
    validation_data=(aoa_val_scaled, cp_val_flat),
    epochs=200,
    batch_size=4,
    callbacks=[early_stop],
    verbose=0 
)

model_A.save(os.path.join(MODELS_DIR, "model_A_full_curve_structured.keras"))
print("Model A trained and saved")

# Architecture B: Pointwise output
print("\nTraining Model B: Predicting one coordinate at a time (Pointwise approach)")

def make_point_dataset(aoa_cases, cp_cases, grid):
    X_pts, y_pts = [], []
    for i in range(len(aoa_cases)):
        current_aoa = aoa_cases[i, 0]
        # Upper surface
        for j, x in enumerate(grid):
            X_pts.append([current_aoa, x, 1.0])
            y_pts.append(cp_cases[i, j])
        # Lower surface
        for j, x in enumerate(grid):
            X_pts.append([current_aoa, x, 0.0])
            y_pts.append(cp_cases[i, NUM_POINTS + j])
    return np.array(X_pts), np.array(y_pts)

X_train_B, y_train_B = make_point_dataset(aoa_train_scaled, cp_train_flat, x_grid)
X_val_B, y_val_B = make_point_dataset(aoa_val_scaled, cp_val_flat, x_grid)
X_test_B, y_test_B = make_point_dataset(aoa_test_scaled, cp_test_flat, x_grid)

model_B = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])

model_B.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

history_B = model_B.fit(
    X_train_B, y_train_B,
    validation_data=(X_val_B, y_val_B),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

model_B.save(os.path.join(MODELS_DIR, "model_B_pointwise_structured.keras"))
print("Model B trained and saved")

# Evaluation & Comparison (on test set)
pred_A = model_A.predict(aoa_test_scaled, verbose=0)
pred_B_flat = model_B.predict(X_test_B, verbose=0)

pred_B = pred_B_flat.reshape(len(aoa_test_scaled), total_cp_points)

# Compare against cp_test_flat
rmse_A = np.sqrt(mean_squared_error(cp_test_flat, pred_A))
rmse_B = np.sqrt(mean_squared_error(cp_test_flat, pred_B))

print("\nExtrapolation Results (Structured Split > 12 deg)")
print(f"Model A (Vector Output) RMSE:    {rmse_A:.5f}")
print(f"Model B (Pointwise Output) RMSE: {rmse_B:.5f}")

if rmse_A < rmse_B:
    print("\nModel A (Vector) is more accurate")
else:
    print("\nModel B (Pointwise) is more accurate")

# plot
plt.figure(figsize=(10, 5))
plt.bar(['Model A (Vector)', 'Model B (Pointwise)'], [rmse_A, rmse_B], color=['darkred', 'darkorange'])
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Extrapolation Stress Test Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate([rmse_A, rmse_B]):
    plt.text(i, v + (max(rmse_A, rmse_B)*0.02), f"{v:.5f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "architecture_comparison_structured.png"), dpi=300)
print(f"Saved comparison plot to {REPORTS_DIR}")