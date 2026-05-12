import numpy as np
import matplotlib.pyplot as plt
import final_functions as f

"""
Convergence study for the displacement heatmap.

This script:
1. Computes displacement heatmaps for several dt and epsilon values.
2. Compares each heatmap to a reference heatmap.
3. Prints absolute and relative errors.
4. Plots all heatmaps with the same fixed colour scale.
5. Optionally plots the trajectory corresponding to the maximum heatmap value
   for one chosen (dt, epsilon) pair.
"""

np.random.seed(1)

# -------------------------------------------------------------------
# CHOICES OF NUMERICAL METHOD
# -------------------------------------------------------------------
forward_choice = f.composition_euler
backward_choice = f.composition_euler

# -------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------
N = 100                  # Number of steps
M = 10                   # Number of perturbation points on circle
perturbation = True      # Whether to perturb the forward image
plot_max_traj = False    # Set True if you want the max trajectory plot

# Values to test for convergence
dt_values = [0.2, 0.1, 0.05, 0.025]
epsilon_values = [0.04, 0.02, 0.01, 0.005]

# Grid
num_points = 100
x_values = np.linspace(-np.pi, np.pi, num_points)
y_values = np.linspace(-np.pi, np.pi, num_points)
X, Y = np.meshgrid(x_values, y_values)

# -------------------------------------------------------------------
# FUNCTION TO COMPUTE A SINGLE HEATMAP
# -------------------------------------------------------------------
def compute_heatmap(dt, epsilon, N=100, M=10, perturbation=True):
    """
    Compute the displacement heatmap for a given dt and epsilon.

    Parameters
    ----------
    dt : float
        Time step.
    epsilon : float
        Perturbation radius.
    N : int
        Number of steps.
    M : int
        Number of perturbation points on the circle.
    perturbation : bool
        Whether to apply perturbations around the forward image.

    Returns
    -------
    heatmap : 2D numpy array
        Average return distance for each initial point.
    """
    heatmap = np.zeros_like(X, dtype=float)

    for i in range(len(y_values)):
        for j in range(len(x_values)):
            x0 = X[i, j]
            y0 = Y[i, j]

            # Forward image
            x1, y1 = forward_choice(x0, y0, dt, N)

            if perturbation:
                sum_distance = 0.0

                for m in range(M):
                    theta = 2 * np.pi * m / M
                    x1_tilde = x1 + epsilon * np.cos(theta)
                    y1_tilde = y1 + epsilon * np.sin(theta)

                    # Backward approximation
                    x_final, y_final = backward_choice(x1_tilde, y1_tilde, -dt, N)

                    # Distance between initial point and return point
                    distance = np.sqrt((x_final - x0) ** 2 + (y_final - y0) ** 2)
                    sum_distance += distance

                heatmap[i, j] = sum_distance / M

            else:
                x_final, y_final = backward_choice(x1, y1, -dt, N)
                distance = np.sqrt((x_final - x0) ** 2 + (y_final - y0) ** 2)
                heatmap[i, j] = distance

    return heatmap


# -------------------------------------------------------------------
# FUNCTION TO PLOT MAX-TRAJECTORY
# -------------------------------------------------------------------
def plot_max_trajectory(heatmap, dt, epsilon, N=100, M=10):
    """
    Plot the forward trajectory and perturbed backward trajectories
    starting from the point that maximises the heatmap.
    """
    i_max, j_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    x0_max = X[i_max, j_max]
    y0_max = Y[i_max, j_max]

    plt.figure(figsize=(7, 7))

    # Forward trajectory
    x_trajectory = [x0_max]
    y_trajectory = [y0_max]

    x_curr, y_curr = x0_max, y0_max
    for _ in range(N):
        x_curr, y_curr = forward_choice(x_curr, y_curr, dt, 1)
        x_trajectory.append(x_curr)
        y_trajectory.append(y_curr)

    plt.plot(x_trajectory, y_trajectory, label="Forward trajectory")

    # Perturbed points and backward trajectories
    x1_tildes = []
    y1_tildes = []

    for m in range(M):
        theta = 2 * np.pi * m / M
        x1_tilde = x_curr + epsilon * np.cos(theta)
        y1_tilde = y_curr + epsilon * np.sin(theta)

        x1_tildes.append(x1_tilde)
        y1_tildes.append(y1_tilde)

        x_back = [x1_tilde]
        y_back = [y1_tilde]

        xb, yb = x1_tilde, y1_tilde
        for _ in range(N):
            xb, yb = backward_choice(xb, yb, -dt, 1)
            x_back.append(xb)
            y_back.append(yb)

        plt.plot(x_back, y_back)
        plt.scatter(x_back[-1], y_back[-1], marker='x')

    plt.scatter(x0_max, y0_max, label="Initial point")
    plt.scatter(x_curr, y_curr, label="Forward image")
    plt.scatter(x1_tildes, y1_tildes, label="Perturbed points")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Max trajectory for dt={dt}, epsilon={epsilon}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


# -------------------------------------------------------------------
# COMPUTE ALL HEATMAPS
# -------------------------------------------------------------------
heatmaps = {}

print("Computing heatmaps...")
for dt in dt_values:
    for epsilon in epsilon_values:
        print(f"  dt = {dt}, epsilon = {epsilon}")
        heatmaps[(dt, epsilon)] = compute_heatmap(
            dt=dt,
            epsilon=epsilon,
            N=N,
            M=M,
            perturbation=perturbation
        )

# -------------------------------------------------------------------
# REFERENCE HEATMAP
# -------------------------------------------------------------------
# Choose the smallest dt and epsilon as the reference
dt_ref = dt_values[-1]
epsilon_ref = epsilon_values[-1]
H_ref = heatmaps[(dt_ref, epsilon_ref)]

print("\nReference heatmap:")
print(f"  dt_ref = {dt_ref}")
print(f"  epsilon_ref = {epsilon_ref}")

# -------------------------------------------------------------------
# ERROR ANALYSIS
# -------------------------------------------------------------------
print("\nErrors relative to the reference heatmap:")
print("(max absolute error and relative Frobenius error)\n")

errors = {}

for (dt, epsilon), H in heatmaps.items():
    abs_error = np.max(np.abs(H - H_ref))
    rel_error = np.linalg.norm(H - H_ref) / np.linalg.norm(H_ref)
    errors[(dt, epsilon)] = (abs_error, rel_error)

    print(
        f"dt = {dt:>7}, epsilon = {epsilon:>7} | "
        f"max error = {abs_error:.6e}, rel error = {rel_error:.6e}"
    )

# -------------------------------------------------------------------
# SUCCESSIVE DIFFERENCE TESTS
# -------------------------------------------------------------------
print("\nConvergence in dt for fixed epsilon values:")
for epsilon in epsilon_values:
    print(f"\n  Fixed epsilon = {epsilon}")
    for k in range(len(dt_values) - 1):
        dt1 = dt_values[k]
        dt2 = dt_values[k + 1]
        H1 = heatmaps[(dt1, epsilon)]
        H2 = heatmaps[(dt2, epsilon)]
        diff = np.max(np.abs(H1 - H2))
        print(f"    dt {dt1} -> {dt2} : max difference = {diff:.6e}")

print("\nConvergence in epsilon for fixed dt values:")
for dt in dt_values:
    print(f"\n  Fixed dt = {dt}")
    for k in range(len(epsilon_values) - 1):
        eps1 = epsilon_values[k]
        eps2 = epsilon_values[k + 1]
        H1 = heatmaps[(dt, eps1)]
        H2 = heatmaps[(dt, eps2)]
        diff = np.max(np.abs(H1 - H2))
        print(f"    epsilon {eps1} -> {eps2} : max difference = {diff:.6e}")

# -------------------------------------------------------------------
# FIXED COLOUR SCALE FOR ALL HEATMAPS
# -------------------------------------------------------------------
global_min = min(np.min(H) for H in heatmaps.values())
global_max = max(np.max(H) for H in heatmaps.values())

print("\nFixed colour scale for all plots:")
print(f"  vmin = {global_min}")
print(f"  vmax = {global_max}")

# -------------------------------------------------------------------
# PLOT ALL HEATMAPS WITH SAME SCALE
# -------------------------------------------------------------------
n_rows = len(dt_values)
n_cols = len(epsilon_values)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
fig.suptitle("Displacement heatmaps with fixed colour scale", fontsize=16)

for i, dt in enumerate(dt_values):
    for j, epsilon in enumerate(epsilon_values):
        ax = axes[i, j] if n_rows > 1 and n_cols > 1 else axes[max(i, j)]
        H = heatmaps[(dt, epsilon)]

        im = ax.imshow(
            H,
            extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
            origin="lower",
            aspect="equal",
            cmap="coolwarm",
            vmin=global_min,
            vmax=global_max
        )

        ax.set_title(f"dt={dt}, ε={epsilon}")
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$y_0$")

# One shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
cbar.set_label("Distance")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# -------------------------------------------------------------------
# OPTIONAL: PLOT MAX TRAJECTORY FOR ONE CASE
# -------------------------------------------------------------------
if plot_max_traj:
    # Choose which heatmap to inspect
    dt_traj = dt_ref
    epsilon_traj = epsilon_ref
    plot_max_trajectory(heatmaps[(dt_traj, epsilon_traj)], dt_traj, epsilon_traj, N=N, M=M)