import os
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Extrapolation Stress Test: Integration Drift
# Objective: Train a direct regression MLP for macroscopic aerodynamic forces (Cl, Cd)
# and benchmark it against the numerically integrated 1D-CNN spatial model during extrapolation.

# Setup
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "airfoil_dataset.npz")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("Loading CFD baseline data for macroscopic forces")
data = np.load(DATA_PATH)
X_raw = data["X"]
y_cl = data["y_cl"].reshape(-1, 1)
y_cd = data["y_cd"].reshape(-1, 1)
x_grid = data["x_grid"]

aoa = X_raw[:, 0].reshape(-1, 1)
y_forces = np.hstack((y_cl, y_cd))

# Structured split (Extrapolation)
aoa_flat = aoa.flatten()
train_mask = aoa_flat <= 8
val_mask   = (aoa_flat > 8) & (aoa_flat <= 12)
test_mask  = aoa_flat > 12

X_train, X_val, X_test = aoa[train_mask], aoa[val_mask], aoa[test_mask]
y_train, y_val, y_test = y_forces[train_mask], y_forces[val_mask], y_forces[test_mask]

# Scale inputs and outputs
input_scaler = StandardScaler()
X_train_scaled = input_scaler.fit_transform(X_train)
X_val_scaled   = input_scaler.transform(X_val)
X_test_scaled  = input_scaler.transform(X_test)

output_scaler = StandardScaler()
y_train_scaled = output_scaler.fit_transform(y_train)
y_val_scaled   = output_scaler.transform(y_val)
y_test_scaled  = output_scaler.transform(y_test)

# Serialize scalers
joblib.dump(input_scaler, os.path.join(MODELS_DIR, "cl_cd_input_scaler_structured.pkl"))
joblib.dump(output_scaler, os.path.join(MODELS_DIR, "cl_cd_output_scaler_structured.pkl"))

# Architecture 1: Direct Global Force MLP
print("\nInitializing Direct Cl/Cd MLP architecture")
mlp_model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="linear") 
])

mlp_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

early_stop = keras.callbacks.EarlyStopping(
    patience=50,
    restore_best_weights=True,
    monitor="val_loss"
)

history = mlp_model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=500,
    batch_size=8,
    callbacks=[early_stop],
    verbose=0
)

mlp_model.save(os.path.join(MODELS_DIR, "cl_cd_direct_model_structured.keras"))
print("Direct MLP training complete.")

# Evaluate MLP on Test Set
mlp_pred_scaled = mlp_model.predict(X_test_scaled, verbose=0)
mlp_pred_real = output_scaler.inverse_transform(mlp_pred_scaled)

rmse_cl_mlp = np.sqrt(mean_squared_error(y_test[:, 0], mlp_pred_real[:, 0]))
rmse_cd_mlp = np.sqrt(mean_squared_error(y_test[:, 1], mlp_pred_real[:, 1]))

# Architecture 2: Spatial 1D-CNN Integration
print("\nLoading Spatial 1D-CNN to evaluate Integration Drift")
try:
    cnn_model = keras.models.load_model(os.path.join(MODELS_DIR, "cnn_cp_model_structured.keras"))
    cnn_in_scaler = joblib.load(os.path.join(MODELS_DIR, "cnn_input_scaler_structured.pkl"))
    
    X_test_cnn_scaled = cnn_in_scaler.transform(X_test)
    cnn_cp_pred = cnn_model.predict(X_test_cnn_scaled, verbose=0)
    
    cl_integrated = []
    
    for i in range(cnn_cp_pred.shape[0]):
        cp_upper = cnn_cp_pred[i, :, 0]
        cp_lower = cnn_cp_pred[i, :, 1]
        
        cn_val = np.trapz(y=(cp_lower - cp_upper), x=x_grid)
        
        aoa_rad = np.radians(X_test[i, 0])
        cl_val = cn_val * np.cos(aoa_rad)
        
        cl_integrated.append(cl_val)
        
    cl_integrated = np.array(cl_integrated)
    rmse_cl_cnn = np.sqrt(mean_squared_error(y_test[:, 0], cl_integrated))
    cnn_available = True

except Exception as e:
    print(f"CNN Integration skipped: {e}")
    cnn_available = False

# Final Benchmark Report
print("\nExtrapolation Benchmark Summary (AoA > 12 deg)")
print(f"Direct MLP Surrogate Lift RMSE : {rmse_cl_mlp:.5f}")
print(f"Direct MLP Surrogate Drag RMSE : {rmse_cd_mlp:.5f}")

if cnn_available:
    print(f"Integrated 1D-CNN Lift RMSE : {rmse_cl_cnn:.5f}")

# Plot
sort_indices = np.argsort(aoa.flatten())
aoa_sorted = aoa[sort_indices]
y_forces_sorted = y_forces[sort_indices]

X_all_scaled = input_scaler.transform(aoa_sorted)
mlp_all_scaled = mlp_model.predict(X_all_scaled, verbose=0)
mlp_all_real = output_scaler.inverse_transform(mlp_all_scaled)

plt.figure(figsize=(12, 5))

# Plot 1: Lift
plt.subplot(1, 2, 1)
plt.plot(aoa_sorted, y_forces_sorted[:, 0], 'k-', linewidth=2, label="CFD Truth (Cl)")
plt.plot(aoa_sorted, mlp_all_real[:, 0], 'b--', linewidth=2, label="Direct MLP")
plt.axvline(x=8, color='gray', linestyle=':', label="Training Boundary")
plt.xlabel("Angle of Attack (Degrees)")
plt.ylabel("Lift Coefficient ($C_l$)")
plt.title("Lift Extrapolation ($C_l$)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Drag
plt.subplot(1, 2, 2)
plt.plot(aoa_sorted, y_forces_sorted[:, 1], 'k-', linewidth=2, label="CFD Truth (Cd)")
plt.plot(aoa_sorted, mlp_all_real[:, 1], 'r--', linewidth=2, label="Direct MLP")
plt.axvline(x=8, color='gray', linestyle=':', label="Training Boundary")
plt.xlabel("Angle of Attack (Degrees)")
plt.ylabel("Drag Coefficient ($C_d$)")
plt.title("Drag Extrapolation ($C_d$)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "global_forces_surrogate_structured.png"), dpi=300)
print(f"\nSaved force prediction plots to {REPORTS_DIR}")
plt.show()