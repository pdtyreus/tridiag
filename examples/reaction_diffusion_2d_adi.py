"""2D Reaction-Diffusion Simulation (Gray-Scott Model) via ADI.

This example simulates morphogenetic Turing pattern formation using the
coupled non-linear Gray-Scott reaction-diffusion equations:

    dU/dt = Du * (d2U/dx2 + d2U/dy2) - U * V^2 + F(y) * (1 - U)
    dV/dt = Dv * (d2V/dx2 + d2V/dy2) + U * V^2 - (F(y) + k) * V

Where:
    U(x, y, t) : Chemical substrate / food concentration
    V(x, y, t) : Autocatalytic morphogen concentration
    Du, Dv     : Diffusion coefficients (Du > Dv)
    F(y)       : Feed rate of U (varies along the vertical axis y)
    k          : Kill / decay rate of V

Numerical Scheme (Peaceman-Rachford ADI):
    1. Half-step (t -> t + dt/2): Implicit diffusion in x, explicit in y, plus reaction.
       Solves batches of Ny tridiagonal systems of size Nx along rows using
       `tridiag.solve(..., axis=1)`.
    2. Full-step (t + dt/2 -> t + dt): Implicit diffusion in y, explicit in x, plus reaction.
       Solves batches of Nx tridiagonal systems of size Ny along columns using
       `tridiag.solve(..., axis=0)`.

Asymmetry Along the Second Axis:
    A linear spatial gradient in the feed rate F(y) along the vertical direction
    drives a morphological transition across the Pearson phase space:
    - Low y (smaller F): Isolated self-replicating spots (solitons)
    - Mid y (medium F): Branching cellular rings and worm-like stripes
    - High y (larger F): Interconnected labyrinthine ribbons
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import tridiag

# --- Physical & Numerical Parameters ---
Nx, Ny = 240, 150  # Interior grid resolution
dx, dy = 1.0, 1.0  # Spatial grid spacing
dt = 1.0  # Time step size
steps = 2500  # Total time steps

# Diffusion rates (Du > Dv is necessary for Turing instability)
du = 0.16
dv = 0.08

# ADI diffusion Courant numbers for U and V
sx_u = du * dt / (2.0 * dx**2)
sy_u = du * dt / (2.0 * dy**2)
sx_v = dv * dt / (2.0 * dx**2)
sy_v = dv * dt / (2.0 * dy**2)

# --- Reaction Parameters with Vertical Asymmetry ---
# A linear gradient in feed rate F along y drives pattern differentiation:
# Bottom: F ~ 0.028 (isolated dividing spots)
# Top:    F ~ 0.056 (dense meandering labyrinth)
y_norm = np.linspace(0.0, 1.0, Ny)[:, None]
f_param = 0.028 + 0.028 * y_norm
k_param = 0.062

# --- Construct Tridiagonal Diagonals (Zero-flux Neumann Boundaries) ---
# Homogeneous diffusion in x (axis 1): 1D diagonals of size Nx - 1, Nx, Nx - 1
ax_u = np.full(Nx - 1, -sx_u)
bx_u = np.full(Nx, 1.0 + 2.0 * sx_u)
bx_u[0] = 1.0 + sx_u  # Neumann boundary at left wall (x = 0)
bx_u[-1] = 1.0 + sx_u  # Neumann boundary at right wall (x = Lx)
cx_u = np.full(Nx - 1, -sx_u)

ax_v = np.full(Nx - 1, -sx_v)
bx_v = np.full(Nx, 1.0 + 2.0 * sx_v)
bx_v[0] = 1.0 + sx_v
bx_v[-1] = 1.0 + sx_v
cx_v = np.full(Nx - 1, -sx_v)

# Homogeneous diffusion in y (axis 0): 1D diagonals of size Ny - 1, Ny, Ny - 1
ay_u = np.full(Ny - 1, -sy_u)
by_u = np.full(Ny, 1.0 + 2.0 * sy_u)
by_u[0] = 1.0 + sy_u  # Neumann boundary at bottom wall (y = 0)
by_u[-1] = 1.0 + sy_u  # Neumann boundary at top wall (y = Ly)
cy_u = np.full(Ny - 1, -sy_u)

ay_v = np.full(Ny - 1, -sy_v)
by_v = np.full(Ny, 1.0 + 2.0 * sy_v)
by_v[0] = 1.0 + sy_v
by_v[-1] = 1.0 + sy_v
cy_v = np.full(Ny - 1, -sy_v)

# --- Initial State ---
# Ambient state: U = 1, V = 0 everywhere
u_field = np.ones((Ny, Nx))
v_field = np.zeros((Ny, Nx))

# Seed grid of localized morphogen disturbances to trigger Turing morphogenesis
rng = np.random.default_rng(42)
for cy_pt in range(25, Ny - 20, 25):
    for cx_pt in range(25, Nx - 20, 30):
        jx = cx_pt + int(rng.integers(-5, 6))
        jy = cy_pt + int(rng.integers(-4, 5))
        r = 6
        u_field[jy - r : jy + r, jx - r : jx + r] = 0.50
        v_field[jy - r : jy + r, jx - r : jx + r] = 0.25

# Small background noise to break spatial symmetry
u_field += 0.01 * rng.standard_normal(u_field.shape)
v_field += 0.01 * rng.standard_normal(v_field.shape)

# Snapshot tracking
snapshot_steps = [400, 1000, 2500]
snapshots = {}

total_solves = steps * 2 * (Ny + Nx) * 2  # 2 half-steps, (Ny + Nx) systems, 2 species
print(f"Mesh: {Nx} x {Ny} nodes")
print(f"Total tridiagonal systems to solve: {total_solves:,}")
print("\nRunning 2D Reaction-Diffusion ADI simulation...")

# --- Time Integration Loop ---
for step in tqdm(range(1, steps + 1)):
    # Local non-linear reaction rates
    uvv = u_field * v_field * v_field
    r_u = -uvv + f_param * (1.0 - u_field)
    r_v = uvv - (f_param + k_param) * v_field

    # --- Half-step 1: Implicit in x, Explicit in y ---
    # Explicit diffusion along y with Neumann boundaries
    d2y_u = np.empty_like(u_field)
    d2y_u[1:-1, :] = u_field[:-2, :] - 2.0 * u_field[1:-1, :] + u_field[2:, :]
    d2y_u[0, :] = u_field[1, :] - u_field[0, :]
    d2y_u[-1, :] = u_field[-2, :] - u_field[-1, :]

    d2y_v = np.empty_like(v_field)
    d2y_v[1:-1, :] = v_field[:-2, :] - 2.0 * v_field[1:-1, :] + v_field[2:, :]
    d2y_v[0, :] = v_field[1, :] - v_field[0, :]
    d2y_v[-1, :] = v_field[-2, :] - v_field[-1, :]

    rhs1_u = u_field + sy_u * d2y_u + 0.5 * dt * r_u
    rhs1_v = v_field + sy_v * d2y_v + 0.5 * dt * r_v

    # Solve across rows along axis 1 (Ny systems of size Nx each)
    u_half = tridiag.solve(ax_u, bx_u, cx_u, rhs1_u, axis=1)
    v_half = tridiag.solve(ax_v, bx_v, cx_v, rhs1_v, axis=1)

    # Reaction rates at intermediate state
    uvv_half = u_half * v_half * v_half
    r_u_half = -uvv_half + f_param * (1.0 - u_half)
    r_v_half = uvv_half - (f_param + k_param) * v_half

    # --- Half-step 2: Implicit in y, Explicit in x ---
    # Explicit diffusion along x with Neumann boundaries
    d2x_u = np.empty_like(u_half)
    d2x_u[:, 1:-1] = u_half[:, :-2] - 2.0 * u_half[:, 1:-1] + u_half[:, 2:]
    d2x_u[:, 0] = u_half[:, 1] - u_half[:, 0]
    d2x_u[:, -1] = u_half[:, -2] - u_half[:, -1]

    d2x_v = np.empty_like(v_half)
    d2x_v[:, 1:-1] = v_half[:, :-2] - 2.0 * v_half[:, 1:-1] + v_half[:, 2:]
    d2x_v[:, 0] = v_half[:, 1] - v_half[:, 0]
    d2x_v[:, -1] = v_half[:, -2] - v_half[:, -1]

    rhs2_u = u_half + sx_u * d2x_u + 0.5 * dt * r_u_half
    rhs2_v = v_half + sx_v * d2x_v + 0.5 * dt * r_v_half

    # Solve across columns along axis 0 (Nx systems of size Ny each)
    u_field = tridiag.solve(ay_u, by_u, cy_u, rhs2_u, axis=0)
    v_field = tridiag.solve(ay_v, by_v, cy_v, rhs2_v, axis=0)

    if step in snapshot_steps:
        snapshots[step] = np.copy(v_field)

# --- Visualization ---
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.4])

# Top Row: Pattern evolution snapshots
for idx, s in enumerate(snapshot_steps):
    ax_snap = fig.add_subplot(gs[0, idx])
    im_snap = ax_snap.imshow(snapshots[s], origin="lower", cmap="magma", vmin=0.0, vmax=0.45)
    ax_snap.set_title(f"Step {s} (t = {s * dt:.0f})")
    ax_snap.set_xlabel("x")
    ax_snap.set_ylabel("y")
    plt.colorbar(im_snap, ax=ax_snap, fraction=0.03, pad=0.04)

# Bottom Left: High-resolution view of fully differentiated Turing patterns
ax_main = fig.add_subplot(gs[1, :2])
im_main = ax_main.imshow(snapshots[steps], origin="lower", cmap="magma", vmin=0.0, vmax=0.45)
ax_main.set_title("Fully Developed Turing Patterns: Spots to Labyrinths (Morphogen V)")
ax_main.set_xlabel("x (grid units)")
ax_main.set_ylabel("y (grid units)")

# Annotate morphological zones along vertical axis
ax_main.text(
    10,
    130,
    "Labyrinthine Ribbons (High F)",
    color="white",
    fontsize=10,
    weight="bold",
    bbox={"boxstyle": "round,pad=0.3", "fc": "black", "alpha": 0.6},
)
ax_main.text(
    10,
    75,
    "Cellular Worms & Rings (Mid F)",
    color="white",
    fontsize=10,
    weight="bold",
    bbox={"boxstyle": "round,pad=0.3", "fc": "black", "alpha": 0.6},
)
ax_main.text(
    10,
    15,
    "Soliton Dots / Spots (Low F)",
    color="white",
    fontsize=10,
    weight="bold",
    bbox={"boxstyle": "round,pad=0.3", "fc": "black", "alpha": 0.6},
)
plt.colorbar(im_main, ax=ax_main, fraction=0.02, pad=0.02, label="Morphogen V Concentration")

# Bottom Right: Vertical profile along y showing pattern transitions
ax_prof = fig.add_subplot(gs[1, 2])
y_coords = np.arange(Ny)
x_slice = Nx // 2
ax_prof.plot(snapshots[steps][:, x_slice], y_coords, "r-", label=f"V at x = {x_slice}")
ax_prof.set_xlabel("Morphogen V")
ax_prof.set_ylabel("y (vertical position)")
ax_prof.set_title("Vertical Profile V(y)")
ax_prof.grid(True, linestyle="--", alpha=0.5)

# Secondary axis to display feed rate F(y)
ax_f = ax_prof.twiny()
ax_f.plot(f_param[:, 0], y_coords, "b--", alpha=0.7, label="Feed rate F(y)")
ax_f.set_xlabel("Feed Rate F", color="blue")
ax_f.tick_params(axis="x", labelcolor="blue")

ax_prof.legend(loc="lower right")
ax_f.legend(loc="upper right")

plt.tight_layout()
plt.show()
