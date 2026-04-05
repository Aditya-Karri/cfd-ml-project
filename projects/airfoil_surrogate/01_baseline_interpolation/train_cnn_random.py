import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Setup & Load
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "airfoil_dataset.npz")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

data = np.load(DATA_PATH)
X_raw = data["X"]
y_cp = data["y_cp"]
x_grid = data["x_grid"]

aoa = X_raw[:, 0].reshape(-1, 1)
NUM_POINTS = y_cp.shape[1] // 2  # Dynamically calculates spatial points per surface (e.g., 150)

print(f"Loaded {len(aoa)} cases. Cp vector shape: {y_cp.shape}")

# Split before scaling to prevent data leakage
aoa_train, aoa_val, cp_train_flat, cp_val_flat = train_test_split(
    aoa, y_cp, test_size=0.2, random_state=SEED
)

# Scale inputs (Output scaling omitted as network handles Cp ranges well)
input_scaler = StandardScaler()
aoa_train_scaled = input_scaler.fit_transform(aoa_train)
aoa_val_scaled = input_scaler.transform(aoa_val)

# Save scaler
joblib.dump(input_scaler, os.path.join(MODELS_DIR, "cnn_input_scaler_random.pkl"))

# Reshape targets into spatial 1D-CNN format: (Batch, Spatial_Grid, Channels) ---> (N, 100, 2)
def format_spatial_channels(cp_flat):
    upper = cp_flat[:, :NUM_POINTS]
    lower = cp_flat[:, NUM_POINTS:]
    return np.stack((upper, lower), axis=-1)

cp_train = format_spatial_channels(cp_train_flat)
cp_val = format_spatial_channels(cp_val_flat)

# Build 1D-CNN Architecture
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),

    # Dense projection to generate latent spatial features
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(NUM_POINTS * 32, activation="relu"),

    # Reshape into a 1D spatial feature map
    tf.keras.layers.Reshape((NUM_POINTS, 32)),

    # Convolutional sequence to smooth and refine the pressure curves
    tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
    tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
    
    # Output: 2 continuous channels (Upper & Lower surfaces)
    tf.keras.layers.Conv1D(2, kernel_size=3, padding="same", activation="linear")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# Training
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
    batch_size=4,
    callbacks=[early_stop],
    verbose=0
)
print("Training complete")

# Evaluation & Saving
pred = model.predict(aoa_val_scaled)

# Flatten for physical RMSE comparison
pred_flat = pred.reshape(pred.shape[0], -1)
cp_val_flat_true = cp_val.reshape(cp_val.shape[0], -1)

rmse = np.sqrt(np.mean((pred_flat - cp_val_flat_true)**2))
print(f"\nValidation RMSE: {rmse:.6f}")

model_save_path = os.path.join(MODELS_DIR, "cnn_cp_model_random.keras")
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")

# Plotting & Exporting
# 1. Training Curves
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss (MSE)")
plt.plot(history.history["val_loss"], label="Val Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("1D-CNN Training Convergence")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "cnn_training_curves_random.png"), dpi=300)
plt.show()

# 2. Cp Distribution (Sample)
sample_idx = 0
aoa_sample = aoa_val[sample_idx][0]

plt.figure(figsize=(10,6))
plt.plot(x_grid, cp_val[sample_idx, :, 0], 'b-', linewidth=2, label="CFD Upper")
plt.plot(x_grid, pred[sample_idx, :, 0], 'r--', linewidth=2, label="CNN Upper")
plt.plot(x_grid, cp_val[sample_idx, :, 1], 'g-', linewidth=2, label="CFD Lower")
plt.plot(x_grid, pred[sample_idx, :, 1], 'orange', linestyle='--', linewidth=2, label="CNN Lower")

plt.gca().invert_yaxis()
plt.xlabel("x/c", fontsize=12)
plt.ylabel("Pressure Coefficient (Cp)", fontsize=12)
plt.title(f"1D-CNN Spatial Surrogate Prediction (AoA = {aoa_sample:.2f} degree)", fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "cnn_cp_prediction_random.png"), dpi=300)
plt.show()