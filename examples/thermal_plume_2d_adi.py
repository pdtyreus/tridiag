"""2D Thermal Plume Simulation via Alternating Direction Implicit (ADI).

This example simulates 2D advection-diffusion of heat from a localized,
off-center continuous heat source placed within a laminar channel flow:

    dT/dt + u(y) * dT/dx = alpha * (d2T/dx2 + d2T/dy2) + Q(x, y)

Where:
    T(x, y, t) : Temperature field (°C)
    u(y)       : Parabolic laminar velocity profile (m/s)
    alpha      : Thermal diffusivity constant (m^2/s)
    Q(x, y)    : Continuous localized heat source (°C/s)

Numerical Scheme (Peaceman-Rachford ADI):
    1. Half-step (t -> t + dt/2): Implicit in x, explicit in y.
       Requires solving a batch of Ny tridiagonal systems of size Nx along
       the horizontal rows using `tridiag.solve(..., axis=1)`.
       Because of advection, these systems have non-symmetric diagonals (a != c).
    2. Full-step (t + dt/2 -> t + dt): Implicit in y, explicit in x.
       Requires solving a batch of Nx tridiagonal systems of size Ny along
       the vertical columns using `tridiag.solve(..., axis=0)`.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import tridiag

# --- Physical & Numerical Parameters ---
Lx, Ly = 2.0, 1.0  # Domain dimensions in meters (length, height)
Nx, Ny = 200, 100  # Number of interior grid points in x and y
dx = Lx / (Nx + 1)
dy = Ly / (Ny + 1)

x = np.linspace(dx, Lx - dx, Nx)
y = np.linspace(dy, Ly - dy, Ny)
X, Y = np.meshgrid(x, y)

alpha = 0.005  # Thermal diffusivity (m^2/s)
u_max = 0.8  # Peak centerline flow velocity (m/s)
dt = 0.015  # Time step size (seconds)
steps = 300  # Total time steps (physical time = 4.5s)
T_ambient = 20.0  # Ambient / inlet temperature (°C)

# --- Laminar Flow & Heat Source ---
# Parabolic channel velocity profile u(y), no-slip at y=0 and y=Ly
u_y = 4.0 * u_max * (y * (Ly - y)) / (Ly**2)

# Off-center Gaussian heat source (e.g. heated component at y=0.35m)
x0, y0 = 0.4, 0.35
sigma = 0.035
Q = 250.0 * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma**2))

# --- ADI Dimensionless Coefficients ---
# Diffusion numbers
sx = alpha * dt / (2.0 * dx**2)
sy = alpha * dt / (2.0 * dy**2)

# Advection Courant numbers (vary vertically across rows)
cx = (u_y * dt / (4.0 * dx))[:, None]  # Shape (Ny, 1)

# Check cell Péclet number for numerical stability with central differences
pe_max = np.max(u_y) * dx / alpha
print(f"Mesh resolution: {Nx} x {Ny} interior nodes")
print(f"Max grid Péclet number: {pe_max:.2f} (<= 2 ensures non-oscillatory central differencing)")

# --- Construct Tridiagonal Matrix Coefficients ---
# Step 1: Horizontal sweep along axis 1 (size Nx, batch Ny)
# Non-symmetric system: lower diagonal != upper diagonal due to advection
ax = np.full((Ny, Nx - 1), -(sx + cx))
bx = np.full((Ny, Nx), 1.0 + 2.0 * sx)
# Zero-gradient (Neumann) outflow at x = Lx absorbed into main diagonal:
bx[:, -1] = 1.0 + sx + cx[:, 0]
cx_diag = np.full((Ny, Nx - 1), -(sx - cx))

# Step 2: Vertical sweep along axis 0 (size Ny, batch Nx)
# Homogeneous diffusion across columns (1D broadcastable diagonals)
ay = np.full(Ny - 1, -sy)
by = np.full(Ny, 1.0 + 2.0 * sy)
cy = np.full(Ny - 1, -sy)

# --- State Initialization ---
T = np.full((Ny, Nx), T_ambient)

# Snapshots for visualization
snapshot_steps = [30, 90, 300]
snapshots = {}

# --- Time Integration Loop (Peaceman-Rachford ADI) ---
print("\nRunning 2D ADI advection-diffusion simulation...")
for step in tqdm(range(1, steps + 1)):
    # --- Half-step 1: Implicit in x, Explicit in y ---
    # Explicit diffusion operator in y: sy * (T_{j-1} - 2*T_j + T_{j+1})
    d2y_t = np.empty_like(T)
    d2y_t[1:-1, :] = T[:-2, :] - 2.0 * T[1:-1, :] + T[2:, :]
    d2y_t[0, :] = T_ambient - 2.0 * T[0, :] + T[1, :]
    d2y_t[-1, :] = T[-2, :] - 2.0 * T[-1, :] + T_ambient

    rhs1 = T + sy * d2y_t + 0.5 * dt * Q
    # Inflow boundary at x = 0 (Dirichlet)
    rhs1[:, 0] += (sx + cx[:, 0]) * T_ambient

    # Batch solve along axis 1 (Ny independent tridiagonal systems of size Nx)
    T_half = tridiag.solve(ax, bx, cx_diag, rhs1, axis=1)

    # --- Half-step 2: Implicit in y, Explicit in x ---
    # Explicit advection-diffusion operator in x:
    # (sx + cx)*T_{i-1} - 2*sx*T_i + (sx - cx)*T_{i+1}
    op_x = np.empty_like(T_half)
    op_x[:, 1:-1] = (
        (sx + cx) * T_half[:, :-2] - 2.0 * sx * T_half[:, 1:-1] + (sx - cx) * T_half[:, 2:]
    )
    op_x[:, 0] = (
        (sx + cx[:, 0]) * T_ambient - 2.0 * sx * T_half[:, 0] + (sx - cx[:, 0]) * T_half[:, 1]
    )
    # Neumann outflow condition at x = Lx
    op_x[:, -1] = (sx + cx[:, 0]) * T_half[:, -2] - (sx + cx[:, 0]) * T_half[:, -1]

    rhs2 = T_half + op_x + 0.5 * dt * Q
    # Dirichlet wall boundaries at y = 0 and y = Ly
    rhs2[0, :] += sy * T_ambient
    rhs2[-1, :] += sy * T_ambient

    # Batch solve along axis 0 (Nx independent tridiagonal systems of size Ny)
    T = tridiag.solve(ay, by, cy, rhs2, axis=0)

    if step in snapshot_steps:
        snapshots[step] = np.copy(T)

# --- Visualization ---
fig = plt.figure(figsize=(12, 8))
# gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

# Top panel: 2D Temperature Field with streamlines and isotherms
# ax0 = fig.add_subplot(gs[0, :])
ax0 = fig.add_subplot()
im = ax0.imshow(
    snapshots[steps],
    origin="lower",
    extent=[0, Lx, 0, Ly],
    cmap="magma",
    aspect="equal",
    vmin=T_ambient,
)
cbar = plt.colorbar(im, ax=ax0, fraction=0.02, pad=0.02)
cbar.set_label("Temperature (°C)")

# Overlay isotherm contour lines
cs = ax0.contour(
    X,
    Y,
    snapshots[steps],
    levels=np.linspace(22, 45, 8),
    colors="white",
    alpha=0.35,
    linewidths=0.8,
)
ax0.clabel(cs, inline=True, fontsize=8, fmt="%.0f°C")

# --- Flow Velocity Visualization (Parabolic Profile) ---
# Inlet parabolic velocity profile curve
x_inlet = 0.04
prof_scale = 0.22 / u_max
ax0.plot(x_inlet + u_y * prof_scale, y, color="cyan", linewidth=2.0, label="Parabolic Profile u(y)")
ax0.plot([x_inlet, x_inlet], [0, Ly], color="cyan", linestyle="--", alpha=0.5)

# Horizontal streamlines with speed-dependent thickness and parabolic arrowheads
y_lines = y[4::9]
u_lines = 4.0 * u_max * (y_lines * (Ly - y_lines)) / (Ly**2)

for y_val, u_val in zip(y_lines, u_lines, strict=False):
    lw = 0.6 + 1.8 * (u_val / u_max)
    alpha = 0.35 + 0.45 * (u_val / u_max)
    ax0.plot([x_inlet + u_val * prof_scale, Lx], [y_val, y_val], color="cyan", lw=lw, alpha=alpha)
    for x_front in [0.75, 1.45]:
        x_arrow = x_front + 0.35 * (u_val / u_max)
        ax0.annotate(
            "",
            xy=(x_arrow + 0.04, y_val),
            xytext=(x_arrow, y_val),
            arrowprops={"arrowstyle": "->", "color": "cyan", "lw": 1.4, "mutation_scale": 11},
        )

# Dotted parabolic wavefronts connecting the arrowheads across the channel
for x_front in [0.75, 1.45]:
    ax0.plot(x_front + 0.35 * (u_y / u_max), y, color="cyan", linestyle=":", lw=0.9, alpha=0.5)

ax0.plot(x0, y0, "r*", markersize=14, markeredgecolor="white", label=f"Heat Source ({x0}m, {y0}m)")
ax0.set_xlim(0, Lx)
ax0.set_ylim(0, Ly)
ax0.set_title("2D Thermal Plume via Alternating Direction Implicit (ADI) Solver (Steady State)")
ax0.set_xlabel("x - Streamwise position (m)")
ax0.set_ylabel("y - Channel height (m)")
ax0.legend(loc="upper right")

plt.tight_layout()
plt.show()
