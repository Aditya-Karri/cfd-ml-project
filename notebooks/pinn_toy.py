# PINN: Solve u''(x) + u(x) = 0
# True solution: u(x) = sin(x)
# Boundary conditions:
# u(0) = 0
# u(pi/2) = 1

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Setup & Domain
tf.random.set_seed(42)
np.random.seed(42)

pi = np.pi

# Collocation points (interior physics points where the ODE must be true)
N_col = 500
x_col = np.random.uniform(0, pi/2, (N_col, 1))
x_col = tf.convert_to_tensor(x_col, dtype=tf.float32)

# Boundary condition points
x_bc0 = tf.constant([[0.0]], dtype=tf.float32)
x_bc1 = tf.constant([[pi/2]], dtype=tf.float32)

# Build the PINN
# tanh is in use because ReLU's second derivative is 0, which breaks the physics loss.
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
])

# Define the Physics-Informed Loss
@tf.function 
def compute_loss():
    # Inner tapes for derivatives: u' and u''
    with tf.GradientTape() as tape2:
        tape2.watch(x_col)
        with tf.GradientTape() as tape1:
            tape1.watch(x_col)
            u = model(x_col)
        du_dx = tape1.gradient(u, x_col)
    d2u_dx2 = tape2.gradient(du_dx, x_col)
    
    # Physics Loss: Residual of u'' + u = 0
    residual = d2u_dx2 + u
    loss_physics = tf.reduce_mean(tf.square(residual))
    
    # Boundary Loss: u(0)=0, u(pi/2)=1
    u0 = model(x_bc0)
    u1 = model(x_bc1)
    loss_bc = tf.reduce_mean(tf.square(u0 - 0.0) + tf.square(u1 - 1.0))
    
    total_loss = loss_physics + loss_bc
    return total_loss, loss_physics, loss_bc

# Training Loop
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
EPOCHS = 5000
loss_history = []

print("Training PINN...")
for epoch in range(EPOCHS):
    # Outer tape tracks weights for the optimizer
    with tf.GradientTape() as tape:
        total_loss, loss_physics, loss_bc = compute_loss()

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Track history
    loss_history.append(float(total_loss))

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Total: {float(total_loss):.6f} | Phys: {float(loss_physics):.6f} | BC: {float(loss_bc):.6f}")

print("Training complete!")

# Evaluation & Plotting
x_test = np.linspace(0, pi/2, 200).reshape(-1, 1)
x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)

u_pred = model(x_test_tf).numpy()
u_true = np.sin(x_test)

# Plot 1: Solution Comparison
plt.figure(figsize=(10,5))
plt.plot(x_test, u_true, 'k-', label="True Solution: sin(x)", linewidth=2)
plt.plot(x_test, u_pred, 'r--', label="PINN Prediction", linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("PINN vs Analytical Solution")
plt.legend()
plt.grid(True, alpha=0.4)
plt.show()

# Plot 2: Physics Residual Check
with tf.GradientTape() as tape2:
    tape2.watch(x_test_tf)
    with tf.GradientTape() as tape1:
        tape1.watch(x_test_tf)
        u = model(x_test_tf)
    du_dx = tape1.gradient(u, x_test_tf)
d2u_dx2 = tape2.gradient(du_dx, x_test_tf)

residual = d2u_dx2 + u

plt.figure(figsize=(10,4))
plt.plot(x_test, residual.numpy(), 'b-')
plt.title("Physics Residual (u'' + u) - Should be near 0")
plt.xlabel("x")
plt.ylabel("Residual")
plt.grid(True, alpha=0.4)
plt.show()

# Plot 3: Loss Convergence
plt.figure(figsize=(10,4))
plt.plot(loss_history, 'g-')
plt.title("Training Loss Convergence")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.yscale('log')
plt.grid(True, alpha=0.4)
plt.show()