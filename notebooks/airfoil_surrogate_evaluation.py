import os
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
import joblib

# Setup & Load Data
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "airfoil_dataset.npz")

data = np.load(DATA_PATH)
X_raw = data["X"]
y_cl = data["y_cl"]
y_cd = data["y_cd"]
x_grid = data["x_grid"]

aoa_features = X_raw[:, 0].reshape(-1, 1)

# Load Day 12 pointwise model and scalers
model_B = keras.models.load_model(os.path.join(PROJECT_ROOT, "models", "model_B_pointwise.keras"))
aoa_scaler = joblib.load(os.path.join(PROJECT_ROOT, "models", "aoa_scaler.pkl"))
cp_scaler = joblib.load(os.path.join(PROJECT_ROOT, "models", "cp_scaler.pkl"))

# Measure Inference Speed (Pointwise Model)
# To predict one full curve, Model B needs 200 rows of [AoA, x/c, surface_flag]
test_aoa = aoa_features[0][0]
test_aoa_scaled = aoa_scaler.transform([[test_aoa]])[0][0]

point_inputs = []
for x in x_grid:
    point_inputs.append([test_aoa_scaled, x, 1.0]) # Upper
for x in x_grid:
    point_inputs.append([test_aoa_scaled, x, 0.0]) # Lower
point_inputs = np.array(point_inputs)

# Warm-up the GPU/CPU to avoid initialization lag in the timing
_ = model_B(point_inputs, training=False)

N_runs = 500
start_time = time.time()
for _ in range(N_runs):
    _ = model_B(point_inputs, training=False)
end_time = time.time()

infer_time_sec = (end_time - start_time) / N_runs

# Read Fluent time dynamically automation report
fluent_time_file = os.path.join(PROJECT_ROOT, "airfoil_cfd_run_time.txt")
fluent_avg_time = 600.0 # fallback if file is missing
if os.path.exists(fluent_time_file):
    with open(fluent_time_file, "r") as f:
        for line in f:
            if "Average CFD Time/Case" in line:
                # Extracts the number from "Average CFD Time/Case (s) : 710.85"
                fluent_avg_time = float(line.split(":")[1].strip())

speedup = fluent_avg_time / infer_time_sec

print("Speed Performance")
print(f"Fluent Avg CFD Time : {fluent_avg_time:.2f} sec")
print(f"Surrogate Inference : {infer_time_sec:.6f} sec (for 200 points)")
print(f"Speedup Factor      : {speedup:,.0f}x\n")


# Uncertainty Ensemble (Cl & Cd)
# We train 3 separate small networks to map AoA -> [Cl, Cd] to estimate uncertainty
y_forces = np.hstack((y_cl.reshape(-1, 1), y_cd.reshape(-1, 1)))

force_scaler = StandardScaler()
y_forces_scaled = force_scaler.fit_transform(y_forces)

aoa_scaled = aoa_scaler.transform(aoa_features)

# Split using same seed to ensure we test on the same holdout cases as before
X_train, X_val, y_train, y_val = train_test_split(
    aoa_scaled, y_forces_scaled, test_size=0.2, random_state=SEED
)

ensemble_models = []
n_models = 3

print("Training Cl/Cd Ensemble (3 Models)")
for i in range(n_models):
    # Slightly vary seed to get different weight initializations
    tf.random.set_seed(SEED + i)
    
    m = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(2) # Output: [Cl, Cd]
    ])
    
    m.compile(optimizer="adam", loss="mse")
    m.fit(X_train, y_train, epochs=250, batch_size=4, verbose=0)
    
    ensemble_models.append(m)
    print(f"  Model {i+1}/3 trained.")

# Evaluate Relative Error and Uncertainty
# Get predictions from all 3 models
raw_preds = [m.predict(X_val, verbose=0) for m in ensemble_models]

# Inverse transform back to real aerodynamic coefficients
real_preds = np.array([force_scaler.inverse_transform(p) for p in raw_preds])

# Calculate Mean and Std Dev across the ensemble
mean_preds = np.mean(real_preds, axis=0)
std_preds = np.std(real_preds, axis=0)

true_forces = force_scaler.inverse_transform(y_val)

# Use Full-Scale Error to prevent divide-by-zero explosions near 0 AoA
cl_max = np.max(np.abs(true_forces[:, 0]))
cd_max = np.max(np.abs(true_forces[:, 1]))

cl_errors = np.abs(mean_preds[:, 0] - true_forces[:, 0]) / cl_max * 100
cd_errors = np.abs(mean_preds[:, 1] - true_forces[:, 1]) / cd_max * 100

print("\n Validation Results (Holdout AoA Cases)")
for i in range(len(X_val)):
    val_aoa = aoa_scaler.inverse_transform([[X_val[i][0]]])[0][0]
    print(f"AoA: {val_aoa:5.1f} deg")
    print(f"  Cl -> CFD: {true_forces[i, 0]:.4f} | ML: {mean_preds[i, 0]:.4f} +/- {std_preds[i, 0]:.4f} (Full-Scale Err: {cl_errors[i]:.2f}%)")
    print(f"  Cd -> CFD: {true_forces[i, 1]:.4f} | ML: {mean_preds[i, 1]:.4f} +/- {std_preds[i, 1]:.4f} (Full-Scale Err: {cd_errors[i]:.2f}%)")
    
print(f"\nAverage Cl Error: {np.mean(cl_errors):.2f}%")
print(f"Average Cd Error: {np.mean(cd_errors):.2f}%")

# Save the Force Scaler
scaler_path = r"D:\cfd-ml-project\models\force_scaler.pkl"
joblib.dump(force_scaler, scaler_path)
print(f"Force scaler saved to: {scaler_path}")

# Save all 3 Ensemble Models
for i, m in enumerate(ensemble_models):
    model_path = r"D:\cfd-ml-project\models\ensemble_clcd_model_{}.keras".format(i+1)
    m.save(model_path)
    print(f"Ensemble Model {i+1} saved to: {model_path}")