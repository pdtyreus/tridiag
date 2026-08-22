import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.optimizers as optim
from tqdm import tqdm
import tridiag

# Heat Transfer Parameter Estimation via Gradient Descent
#
# This example shows how to discover an unknown thermal
# profile along a 1D metal rod (e.g. aluminum with a 
# copper core) using gradient descent and back propagating
# through the differentiable linear solver. This pattern
# is known as physics-informed ML.
# 
# Using the MLX implementation of the solver allows
# the entire process to execute in MLX's computation
# graph instead of delegating the physics simulation
# to the CPU. 

# --- Physical & Numerical Parameters ---
L = 1.0          # Length of rod (meters)
N = 50           # Number of spatial interior grid points
dx = L / (N + 1) # Grid spacing

dt = 0.05        # Time step size (seconds)
steps = 50       # Total time steps to simulate
 

# alpha is the Thermal diffusivity (m^2/s)
# In this simulation we are modeling something like aluminum
# with a higher conducting core. A real profile would have a
# sharp cutoff at the transition betwen the metals. To make this 
# example simpler we instead model a gradual transition profile 
# of alpha through the rod. This profile is what we want
# to recover using parameter estimation.
x = mx.array([(i + 1) * dx for i in range(N)])
true_alpha = 1e-4 * (1.0 + 2.0 * mx.exp(-((x - 0.5)**2) / 0.05))

# The physics simulation

def simulate(alpha_profile):
    """Forward simulation solver wrapped in an MLX function."""
    T = mx.full((N,), 20.0) # Initial condition (20°C)
    T_left, T_right = 100.0, 20.0
    
    for _ in range(steps):
        # Local Courant numbers along the grid
        r = alpha_profile * dt / (dx**2)
        
        # Assemble tridiagonal matrix diagonals
        # For non-uniform alpha, sub/super diagonals scale locally
        a = -r[1:]
        b = 1.0 + 2.0 * r
        c = -r[:-1]
        
        rhs = mx.array(T)
        rhs = rhs.at[0].add(r[0] * T_left)
        rhs = rhs.at[-1].add(r[-1] * T_right)
        
        # Differentiable solve step
        # Specifically use the CR MLX implementation via the main 
        # entry point
        T = tridiag.solve(a, b, c, rhs, method="cr_mlx")
    return T

# Generate synthetic "observed" temperature profile from true 
# physical parameters and add some noise.
T_observed = simulate(true_alpha) + 0.1 * mx.random.normal((N,))

# --- Inverse Problem Optimization Loop ---
# We will parameterize as a radial basis function for this example.
# This is slightly "cheating" in that we would need to know that
# the profile is a curve. However, since it is just an example,
# this makes for better convergence.
#
# Optimize only 3 parameters: baseline, peak height, and width
params = {
    "base": mx.array([1e-4]),
    "amplitude": mx.array([1e-4]),
    "center": mx.array([0.5])
}

def get_alpha(params):
    base = mx.maximum(params["base"], 1e-6)
    amp = mx.maximum(params["amplitude"], 0.0)
    # Reconstruct smooth alpha array across grid
    return base + amp * mx.exp(-((x - params["center"])**2) / 0.05)

def loss_fn(params):
    alpha_profile = get_alpha(params)
    T_pred = simulate(alpha_profile)
    return mx.mean(mx.square(T_pred - T_observed))

# Obtain gradient function directly through the solver using MLX autograd
loss_and_grad_fn = mx.value_and_grad(loss_fn)
optimizer = optim.Adam(learning_rate=1e-5)

print("Starting Parameter Estimation via Backpropagation...")
pbar = tqdm(range(101))
for step in pbar:
    loss, grads = loss_and_grad_fn(params)
    optimizer.update(params, grads)
    mx.eval(params, optimizer.state)
    
    pbar.set_description(f"MSE Loss: {loss.item():.6f}")

# reconstruct the radial basis function into an alpha curve
learned_alpha = params["base"] + params["amplitude"] * mx.exp(-((x - params["center"])**2) / 0.05)
print("\nOptimization complete! Learned alpha successfully recovered.")

# --- Visualization ---
# Convert MLX arrays to standard Python lists for plotting
x_np = x.tolist()
true_alpha_np = true_alpha.tolist()
learned_alpha_np = learned_alpha.tolist()

plt.figure(figsize=(9, 5))
plt.plot(x_np, true_alpha_np, 'g-', linewidth=2, label="True Alpha Profile")
plt.plot(x_np, learned_alpha_np, 'r--', linewidth=2, label="Recovered Alpha Profile")
plt.axhline(y=1e-4, color='gray', linestyle=':', label="Initial Guess")
plt.title("Parameter Estimation: Thermal Diffusivity Profile Recovery")
plt.xlabel("Position along rod (m)")
plt.ylabel("Thermal Diffusivity (m^2/s)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()