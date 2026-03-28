import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Setup & Load Data
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "airfoil_dataset.npz")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

data = np.load(DATA_PATH)
X_raw = data["X"]
y_cp = data["y_cp"]
y_cl = data["y_cl"].reshape(-1, 1)
x_grid = data["x_grid"]

aoa = X_raw[:, 0].reshape(-1, 1)
dx = x_grid[1] - x_grid[0] # Uniform grid spacing

print(f"Loaded {len(X_raw)} cases.")

# Train/Val Split & Scaling
# Split BEFORE scaling to prevent data leakage from val statistics bleeding into train
aoa_train, aoa_val, cp_train, cp_val, cl_train, cl_val = train_test_split(
    aoa, y_cp, y_cl, test_size=0.2, random_state=SEED
)

input_scaler = StandardScaler()
aoa_train_scaled = input_scaler.fit_transform(aoa_train)
aoa_val_scaled = input_scaler.transform(aoa_val)

output_scaler = StandardScaler()
cp_train_scaled = output_scaler.fit_transform(cp_train)
cp_val_scaled = output_scaler.transform(cp_val)

# Save scalers for future demo usage
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
joblib.dump(input_scaler, os.path.join(PROJECT_ROOT, "models", "pinn_airfoil_input_scaler.pkl"))
joblib.dump(output_scaler, os.path.join(PROJECT_ROOT, "models", "pinn_airfoil_output_scaler.pkl"))

# Convert everything to tensors for the custom training loop
X_train_tf = tf.convert_to_tensor(aoa_train_scaled, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(cp_train_scaled, dtype=tf.float32)
cl_train_tf = tf.convert_to_tensor(cl_train, dtype=tf.float32)

X_val_tf = tf.convert_to_tensor(aoa_val_scaled, dtype=tf.float32)
y_val_tf = tf.convert_to_tensor(cp_val_scaled, dtype=tf.float32)
cl_val_tf = tf.convert_to_tensor(cl_val, dtype=tf.float32)

# Build PINN Architecture
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(200) # Output is the full 200 point curve
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Physics constraints
lambda_phys = 0.1 # Weighting factor: how strictly should it obey the physics vs the data?
cp_mean = tf.constant(output_scaler.mean_, dtype=tf.float32)
cp_scale = tf.constant(output_scaler.scale_, dtype=tf.float32)

def inverse_scale_cp(cp_scaled):
    """Brings network predictions back to real aerodynamic values for physics calculations."""
    return cp_scaled * cp_scale + cp_mean

def compute_cl_from_cp(cp_real):
    """Trapezoidal integration of (Cp_lower - Cp_upper) along the chord to find Lift."""
    cp_upper = cp_real[:, :100]
    cp_lower = cp_real[:, 100:]
    
    # Delta Cp
    dcp = cp_lower - cp_upper
    
    # Trapezoidal rule: sum( (f(x_i) + f(x_{i+1}))/2 * dx )
    # Which simplifies to: dx * ( 0.5*f_0 + f_1 + ... + f_{N-1} + 0.5*f_N )
    ends = 0.5 * (dcp[:, 0] + dcp[:, -1])
    mids = tf.reduce_sum(dcp[:, 1:-1], axis=1)
    
    cl_pred = (ends + mids) * dx
    return tf.reshape(cl_pred, (-1, 1))

# Custom Training Loop
@tf.function
def train_step(x, y_true, cl_true):
    with tf.GradientTape() as tape:
        cp_pred_scaled = model(x, training=True)
        
        # 1. Data Loss (MSE in scaled space)
        data_loss = tf.reduce_mean(tf.square(cp_pred_scaled - y_true))
        
        # 2. Physics Loss (MSE of integrated lift in real space)
        cp_pred_real = inverse_scale_cp(cp_pred_scaled)
        cl_pred = compute_cl_from_cp(cp_pred_real)
        phys_loss = tf.reduce_mean(tf.square(cl_pred - cl_true))
        
        # 3. Total Loss
        total_loss = data_loss + (lambda_phys * phys_loss)
        
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return data_loss, phys_loss

epochs = 300
history = {'train_data': [], 'train_phys': [], 'val_data': [], 'val_phys': []}

print("Starting PINN training")
for epoch in range(epochs):
    
    # Train
    data_loss, phys_loss = train_step(X_train_tf, y_train_tf, cl_train_tf)
    
    # Validate
    cp_val_pred_scaled = model(X_val_tf, training=False)
    val_data_loss = tf.reduce_mean(tf.square(cp_val_pred_scaled - y_val_tf))
    
    cp_val_pred_real = inverse_scale_cp(cp_val_pred_scaled)
    cl_val_pred = compute_cl_from_cp(cp_val_pred_real)
    val_phys_loss = tf.reduce_mean(tf.square(cl_val_pred - cl_val_tf))
    
    # Log
    history['train_data'].append(float(data_loss))
    history['train_phys'].append(float(phys_loss))
    history['val_data'].append(float(val_data_loss))
    history['val_phys'].append(float(val_phys_loss))
    
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Train Data: {float(data_loss):.5f} | Train Phys: {float(phys_loss):.5f} | Val Phys: {float(val_phys_loss):.5f}")

print("Training complete")
model.save(os.path.join(PROJECT_ROOT, "models", "pinn_airfoil.keras"))

# Evaluation & Plots
# Loss Curves
plt.figure(figsize=(10,5))
plt.plot(history['train_data'], label="Train Data Loss (Scaled MSE)")
plt.plot(history['val_data'], label="Val Data Loss (Scaled MSE)")
plt.plot(history['train_phys'], '--', label="Train Physics Loss (Cl Error)")
plt.plot(history['val_phys'], '--', label="Val Physics Loss (Cl Error)")
plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("PINN Training Convergence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(REPORTS_DIR, "PINN_Training_Convergence.png"), dpi=300)
plt.show()

# Cp Distribution Check
sample_idx = 0
test_aoa_val = aoa_val[sample_idx][0]

cp_pred_scaled = model(X_val_tf[sample_idx:sample_idx+1], training=False)
cp_pred_real = inverse_scale_cp(cp_pred_scaled).numpy().flatten()
cp_true_real = cp_val[sample_idx].flatten()

plt.figure(figsize=(10,6))
plt.plot(x_grid, cp_true_real[:100], 'k-', linewidth=2, label="CFD Upper")
plt.plot(x_grid, cp_true_real[100:], 'k--', linewidth=2, label="CFD Lower")
plt.plot(x_grid, cp_pred_real[:100], 'r-', linewidth=1.5, label="PINN Upper")
plt.plot(x_grid, cp_pred_real[100:], 'b-', linewidth=1.5, label="PINN Lower")

plt.gca().invert_yaxis()
plt.xlabel("x/c")
plt.ylabel("Pressure Coefficient (Cp)")
plt.title(f"PINN $c_p$ Prediction vs CFD (AoA = {test_aoa_val:.1f} deg)")
plt.savefig(os.path.join(REPORTS_DIR, "pinn_cp_prediction.png"), dpi=300)
plt.legend()
plt.grid(True, alpha=0.4)
plt.show()