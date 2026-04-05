import os
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Physics-Informed Neural Network (PINN) Prototype
# Objective: Evaluate if adding a custom physical loss penalty (Lift Integration) 
# to a standard MLP improves extrapolation fidelity for pressure distributions.

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
y_cl = data["y_cl"].reshape(-1, 1)
x_grid = data["x_grid"]

aoa = X_raw[:, 0].reshape(-1, 1)

NUM_POINTS = y_cp.shape[1] // 2 
total_cp_points = y_cp.shape[1] 
dx = tf.constant(x_grid[1] - x_grid[0], dtype=tf.float32)

print(f"Loaded {len(aoa)} cases. Target shape: {y_cp.shape}")

# Structured split (Extrapolation)
aoa_flat = aoa.flatten()
train_mask = aoa_flat <= 8
val_mask   = (aoa_flat > 8) & (aoa_flat <= 12)
test_mask  = aoa_flat > 12

aoa_train, aoa_val, aoa_test = aoa[train_mask], aoa[val_mask], aoa[test_mask]
cp_train, cp_val, cp_test = y_cp[train_mask], y_cp[val_mask], y_cp[test_mask]
cl_train, cl_val, cl_test = y_cl[train_mask], y_cl[val_mask], y_cl[test_mask]

# Scale inputs (fit only on training data)
input_scaler = StandardScaler()
aoa_train_scaled = input_scaler.fit_transform(aoa_train)
aoa_val_scaled   = input_scaler.transform(aoa_val)
aoa_test_scaled  = input_scaler.transform(aoa_test)

joblib.dump(input_scaler, os.path.join(MODELS_DIR, "pinn_mlp_input_scaler_structured.pkl"))

# Physics Constants for un-scaling inside the TF Graph
AOA_MEAN = tf.constant(input_scaler.mean_[0], dtype=tf.float32)
AOA_STD = tf.constant(input_scaler.scale_[0], dtype=tf.float32)

# Dataset generation
BATCH_SIZE = 16
train_dataset = tf.data.Dataset.from_tensor_slices((
    aoa_train_scaled.astype(np.float32),
    cp_train.astype(np.float32),
    cl_train.astype(np.float32)
)).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    aoa_val_scaled.astype(np.float32),
    cp_val.astype(np.float32),
    cl_val.astype(np.float32)
)).batch(BATCH_SIZE)

# Architecture: Dense PINN
print(f"\nBuilding Dense PINN Architecture ({total_cp_points} outputs)")
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(1,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(total_cp_points, activation="linear")
])

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
LAMBDA_PHYSICS = 0.5 

# Custom Physics-Informed Training Step
@tf.function
def train_step(x_batch, y_cp_batch, y_cl_batch):
    with tf.GradientTape() as tape:
        cp_pred = model(x_batch, training=True)
        data_loss = tf.reduce_mean(tf.square(y_cp_batch - cp_pred))
        
        cp_upper = cp_pred[:, :NUM_POINTS]
        cp_lower = cp_pred[:, NUM_POINTS:]
        
        dcp = cp_lower - cp_upper
        cn_pred = tf.reduce_sum(dcp, axis=1, keepdims=True) * dx
        
        aoa_real = (x_batch * AOA_STD) + AOA_MEAN
        aoa_rad = aoa_real * (np.pi / 180.0)
        cl_pred = cn_pred * tf.cos(aoa_rad)
        
        phys_loss = tf.reduce_mean(tf.square(y_cl_batch - cl_pred))
        total_loss = data_loss + (LAMBDA_PHYSICS * phys_loss)
        
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return data_loss, phys_loss, total_loss

@tf.function
def val_step(x_batch, y_cp_batch, y_cl_batch):
    cp_pred = model(x_batch, training=False)
    data_loss = tf.reduce_mean(tf.square(y_cp_batch - cp_pred))
    
    cp_upper = cp_pred[:, :NUM_POINTS]
    cp_lower = cp_pred[:, NUM_POINTS:]
    
    dcp = cp_lower - cp_upper
    cn_pred = tf.reduce_sum(dcp, axis=1, keepdims=True) * dx
    
    aoa_real = (x_batch * AOA_STD) + AOA_MEAN
    aoa_rad = aoa_real * (np.pi / 180.0)
    cl_pred = cn_pred * tf.cos(aoa_rad)
    
    phys_loss = tf.reduce_mean(tf.square(y_cl_batch - cl_pred))
    total_loss = data_loss + (LAMBDA_PHYSICS * phys_loss)
    return data_loss, phys_loss, total_loss

# Execution
EPOCHS = 350
history = {"train_total_loss": [], "val_total_loss": [], "val_phys_loss": []}

print("\nStarting PINN Optimization")
for epoch in range(EPOCHS):
    t_total = tf.keras.metrics.Mean()
    v_total = tf.keras.metrics.Mean()
    v_phys = tf.keras.metrics.Mean()
    
    for xb, ycp, ycl in train_dataset:
        _, _, tl = train_step(xb, ycp, ycl)
        t_total.update_state(tl)
        
    for xb, ycp, ycl in val_dataset:
        _, pl, tl = val_step(xb, ycp, ycl)
        v_total.update_state(tl)
        v_phys.update_state(pl)
        
    history["train_total_loss"].append(t_total.result().numpy())
    history["val_total_loss"].append(v_total.result().numpy())
    history["val_phys_loss"].append(v_phys.result().numpy())
    
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:03d} | Train Loss: {t_total.result():.4f} | Val Loss: {v_total.result():.4f} | Physics Error: {v_phys.result():.6f}")

model.save(os.path.join(MODELS_DIR, "pinn_mlp_model_structured.keras"))
print("Model saved successfully")

# Plot 1: Convergence
plt.figure(figsize=(10, 6))
plt.plot(history["train_total_loss"], label="Train Total Loss", color='blue', linewidth=2)
plt.plot(history["val_total_loss"], label="Val Total Loss", color='orange', linewidth=2)
plt.plot(history["val_phys_loss"], label="Physics Penalty (Cl Error)", color='green', linestyle=':', linewidth=2)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Dense PINN Training Convergence")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "pinn_mlp_training_convergence.png"), dpi=300)
plt.show()

# Plot 2: Extrapolation Evaluation (Most extreme test case)
sample_idx = -1 
aoa_sample = aoa_test[sample_idx][0]

test_input = aoa_test_scaled[sample_idx].reshape(1, -1)
pred_test = model.predict(test_input, verbose=0)

cp_upper_pred = pred_test[0, :NUM_POINTS]
cp_lower_pred = pred_test[0, NUM_POINTS:]

plt.figure(figsize=(10,6))
plt.plot(x_grid, cp_test[sample_idx, :NUM_POINTS], color='black', linewidth=2.5, label="CFD Upper")
plt.plot(x_grid, cp_test[sample_idx, NUM_POINTS:], color='gray', linewidth=2.5, linestyle='-.', label="CFD Lower")
plt.plot(x_grid, cp_upper_pred, color='blue', linewidth=2, linestyle='--', label="PINN MLP Upper")
plt.plot(x_grid, cp_lower_pred, color='orange', linewidth=2, linestyle='--', label="PINN MLP Lower")

plt.gca().invert_yaxis() 
plt.xlabel("x/c")
plt.ylabel("Pressure Coefficient (Cp)")
plt.title(f"Dense PINN Extrapolation Prediction (AoA = {aoa_sample:.1f} degree)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "pinn_mlp_cp_prediction.png"), dpi=300)
plt.show()