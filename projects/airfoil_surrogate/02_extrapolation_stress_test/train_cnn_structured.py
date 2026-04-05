import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Extrapolation Stress Test
# Objective: Evaluate the 1D-CNN's physical accuracy when extrapolating to 
# high Angles of Attack (>12 deg) using a structured data split.

# setup
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

# Dynamically get the number of points 
NUM_POINTS = y_cp.shape[1] // 2 

#  Structured split (Extrapolation)
aoa_flat = aoa.flatten()
train_mask = aoa_flat <= 8
val_mask   = (aoa_flat > 8) & (aoa_flat <= 12)
test_mask  = aoa_flat > 12

aoa_train, aoa_val, aoa_test = aoa[train_mask], aoa[val_mask], aoa[test_mask]
cp_train_flat, cp_val_flat, cp_test_flat = y_cp[train_mask], y_cp[val_mask], y_cp[test_mask]

# Scale inputs 
input_scaler = StandardScaler()
aoa_train_scaled = input_scaler.fit_transform(aoa_train)
aoa_val_scaled   = input_scaler.transform(aoa_val)
aoa_test_scaled  = input_scaler.transform(aoa_test)

# Save scaler
joblib.dump(input_scaler, os.path.join(MODELS_DIR, "cnn_input_scaler_structured.pkl"))

# data formatting for CNN
def format_spatial_channels(cp_flat):
    upper = cp_flat[:, :NUM_POINTS]
    lower = cp_flat[:, NUM_POINTS:]
    return np.stack((upper, lower), axis=-1)

cp_train = format_spatial_channels(cp_train_flat)
cp_val   = format_spatial_channels(cp_val_flat)
cp_test  = format_spatial_channels(cp_test_flat)

# Build 1D-CNN Architecture
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(NUM_POINTS * 32, activation="relu"),
    tf.keras.layers.Reshape((NUM_POINTS, 32)),
    tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
    tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
    tf.keras.layers.Conv1D(2, kernel_size=3, padding="same", activation="linear")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

# training
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=100,
    restore_best_weights=True,
    monitor="val_loss"
)

print("\nStarting CNN surrogate training")
history = model.fit(
    aoa_train_scaled, cp_train,
    validation_data=(aoa_val_scaled, cp_val),
    epochs=2000,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0 
)
print(f"Training complete. Restored best weights from epoch {early_stop.best_epoch}.")

# Evaluation & Comparison (on test set)
pred_test = model.predict(aoa_test_scaled, verbose=0)
test_flat = pred_test.reshape(pred_test.shape[0], -1)

rmse_test = np.sqrt(np.mean((test_flat - cp_test_flat)**2))
print(f"\n1D-CNN Extrapolation Performance (RMSE): {rmse_test:.5f}")

# Save model
model.save(os.path.join(MODELS_DIR, "cnn_cp_model_structured.keras"))
print("Model saved successfully")

# Plot
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss (MSE)")
plt.plot(history.history["val_loss"], label="Val Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("1D-CNN Training Convergence (Structured Split)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "cnn_training_curves_structured.png"), dpi=300)
plt.show()

# Cp Extrapolation Plot (Using the most extreme Test case)
sample_idx = -1 
aoa_sample = aoa_test[sample_idx][0]

plt.figure(figsize=(10,6))
plt.plot(x_grid, cp_test[sample_idx, :, 0], 'b-', linewidth=2, label="CFD Upper")
plt.plot(x_grid, pred_test[sample_idx, :, 0], 'r--', linewidth=2, label="CNN Upper")
plt.plot(x_grid, cp_test[sample_idx, :, 1], 'g-', linewidth=2, label="CFD Lower")
plt.plot(x_grid, pred_test[sample_idx, :, 1], 'orange', linestyle='--', linewidth=2, label="CNN Lower")

plt.gca().invert_yaxis()
plt.xlabel("x/c", fontsize=12)
plt.ylabel("Pressure Coefficient (Cp)", fontsize=12)
plt.title(f"1D-CNN Extrapolation Prediction (AoA = {aoa_sample:.2f} degree)", fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "cnn_cp_prediction_structured.png"), dpi=300)
plt.show()