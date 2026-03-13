import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import random

# Configuration & Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "airfoil_dataset.npz")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Load Dataset
print("Loading aerodynamic dataset")
data = np.load(DATA_PATH)
X_raw = data["X"]
y_cp = data["y_cp"]
y_cl = data["y_cl"].reshape(-1, 1)
y_cd = data["y_cd"].reshape(-1, 1)
x_grid = data["x_grid"]

aoa = X_raw[:, 0].reshape(-1, 1)
y_forces = np.hstack((y_cl, y_cd))

print(f"Total samples loaded: {aoa.shape[0]}")

# Train/Validation Split & Scaling
# Split executed prior to scaling to enforce strict data isolation
X_train, X_val, y_train, y_val = train_test_split(
    aoa, y_forces, test_size=0.2, random_state=SEED
)

# Fit scalers strictly on training distributions
input_scaler = StandardScaler()
X_train_scaled = input_scaler.fit_transform(X_train)
X_val_scaled = input_scaler.transform(X_val)

output_scaler = StandardScaler()
y_train_scaled = output_scaler.fit_transform(y_train)
y_val_scaled = output_scaler.transform(y_val)

# Direct Cl/Cd MLP Model
print("\nInitializing Direct Cl/Cd MLP architecture...")
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(2, activation="linear") # Outputs: [Cl, Cd]
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
    loss="mse"
)

early_stop = tf.keras.callbacks.EarlyStopping(
    patience=30,
    restore_best_weights=True,
    monitor="val_loss"
)

print("Commencing Direct Model training sequence")
model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=300,
    batch_size=4,
    callbacks=[early_stop],
    verbose=0
)

# Direct Model Evaluation
pred_forces_scaled = model.predict(X_val_scaled, verbose=0)
pred_forces_real = output_scaler.inverse_transform(pred_forces_scaled)

rmse_cl = np.sqrt(mean_squared_error(y_val[:, 0], pred_forces_real[:, 0]))
rmse_cd = np.sqrt(mean_squared_error(y_val[:, 1], pred_forces_real[:, 1]))

# Calculate filtered percentage errors (ignoring near-zero denominators)
cl_range = np.max(y_val[:, 0]) - np.min(y_val[:, 0])
norm_rmse_cl = (rmse_cl / cl_range) * 100

mask = np.abs(y_val[:, 0]) > 0.1
mean_cl_err_direct = np.mean(np.abs(pred_forces_real[mask, 0] - y_val[mask, 0]) / np.abs(y_val[mask, 0])) * 100
mean_cd_err_direct = np.mean(np.abs(pred_forces_real[:, 1] - y_val[:, 1]) / np.abs(y_val[:, 1])) * 100

# Save Direct Model artifacts
os.makedirs(MODELS_DIR, exist_ok=True)
model.save(os.path.join(MODELS_DIR, "cl_cd_direct_model.keras"))
joblib.dump(input_scaler, os.path.join(MODELS_DIR, "cl_cd_input_scaler.pkl"))
joblib.dump(output_scaler, os.path.join(MODELS_DIR, "cl_cd_output_scaler.pkl"))

# Cp-Integrated Cl from 1D-CNN
print("\nLoading 1D-CNN Cp Surrogate for integration comparison")
cnn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cnn_cp_model.keras"))
cnn_in_scaler = joblib.load(os.path.join(MODELS_DIR, "cnn_input_scaler.pkl"))
cnn_out_scaler = joblib.load(os.path.join(MODELS_DIR, "cnn_output_scaler.pkl"))

# Predict Cp using validation AoAs
X_val_scaled_cnn = cnn_in_scaler.transform(X_val)
pred_cp_cnn_scaled = cnn_model.predict(X_val_scaled_cnn, verbose=0)

# CRITICAL FIX: Unpack channels to avoid interleaving the array
cp_upper_scaled = pred_cp_cnn_scaled[:, :, 0]
cp_lower_scaled = pred_cp_cnn_scaled[:, :, 1]

# Horizontally stack to recreate the exact (N, 200) shape the scaler expects
pred_cp_flat_scaled = np.hstack((cp_upper_scaled, cp_lower_scaled))

# Inverse transform to physical reality
pred_cp_flat_real = cnn_out_scaler.inverse_transform(pred_cp_flat_scaled)

# Integrate Cl from Cp using Trapezoidal Rule
cl_integrated = []
for i in range(pred_cp_flat_real.shape[0]):
    # Extract the un-scrambled real physics
    cp_upper_real = pred_cp_flat_real[i, :100]
    cp_lower_real = pred_cp_flat_real[i, 100:]
    
    # Trapezoidal integration of (Cp_lower - Cp_upper) along the chord
    cl_val = np.trapz(y=(cp_lower_real - cp_upper_real), x=x_grid)
    cl_integrated.append(cl_val)

cl_integrated = np.array(cl_integrated)

# Integrated Cl Evaluation
rmse_cl_int = np.sqrt(mean_squared_error(y_val[:, 0], cl_integrated))
norm_rmse_cl_int = (rmse_cl_int / cl_range) * 100
mean_cl_err_int = np.mean(np.abs(cl_integrated[mask] - y_val[mask, 0]) / np.abs(y_val[mask, 0])) * 100

# Final Report
print("     SURROGATE PERFORMANCE SUMMARY      ")
print("1. Direct MLP Prediction (Cl, Cd):")
print(f"   - Normalized RMSE Cl : {norm_rmse_cl:.2f}%")
print(f"   - Mean % Error Cl    : {mean_cl_err_direct:.2f}%")
print(f"   - Mean % Error Cd    : {mean_cd_err_direct:.2f}%")
print("2. 1D-CNN Integrated Prediction (Cl):")
print(f"   - Normalized RMSE Cl : {norm_rmse_cl_int:.2f}%")
print(f"   - Mean % Error Cl    : {mean_cl_err_int:.2f}%")

if mean_cl_err_int < mean_cl_err_direct:
    print("\n The CNN-Integrated model provides higher accuracy for Lift.")
else:
    print("\n The Direct MLP model provides higher accuracy for Lift.")