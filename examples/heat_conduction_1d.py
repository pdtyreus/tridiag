import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import tridiag

# Simulation of 1D coductive heat transfer in a metal rod.
# 
# 1D Heat Equation: du/dt = D * d2u/dx2
# Where:
#   u = temperature
#   t = time
#   x = spatial position
#   D = thermal diffusivity constant


# --- Physical & Numerical Parameters ---
L = 0.5          # Length of rod (meters)
N = 1000         # Number of spatial interior grid points
dx = L / (N + 1) # Grid spacing

alpha = 1e-4     # Thermal diffusivity (m^2/s), e.g., aluminum
dt = 0.5         # Time step size (seconds)
steps = 500      # Total time steps to simulate

# stability parameter
r = alpha * dt / (dx**2)

# --- Initial and Boundary Conditions ---
T_left = 100.0   # Fixed left boundary (C)
T_right = 20.0   # Fixed right boundary (C)

# Interior nodes initialized at 20 C
T = np.full((N,), 20.0)

# --- Construct Tridiagonal Matrix Coefficients for Implicit Euler ---
# System: a[i]*T_{i-1} + b[i]*T_i + c[i]*T_{i+1} = rhs[i]
a = np.full((N - 1,), -r)     # Sub-diagonal
b = np.full((N,), 1.0 + 2*r)  # Main diagonal
c = np.full((N - 1,), -r)     # Super-diagonal

# --- Time Integration Loop ---
history = [T]
history_step = 50

for step in tqdm(range(steps)):
    # Right-hand side is previous state, with boundary terms injected
    rhs = np.array(T)
    rhs[0] += r * T_left
    rhs[-1] += r * T_right
    
    # Solve tridiagonal system for next time step T^{n+1}
    # Easiest is to call `tridiag.solve(a, b, c, rhs)`
    # and let the library choose the implementation. But 
    # for this tiny system we can just use the python solver.
    T = tridiag.solvers.thomas.solve_vanilla(a, b, c, rhs)
    
    if step % history_step == 0:
        history.append(T)

# --- Visualization ---
x = [i * dx for i in range(1, N + 1)]

plt.figure(figsize=(9, 5))
for i, snapshot in enumerate(history):
    plt.plot(x, snapshot, label=f"t = {i * history_step * dt:.0f}s")

plt.title("Heat Conduction in a 1D Rod")
plt.xlabel("Position along rod (m)")
plt.ylabel("Temperature (°C)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()