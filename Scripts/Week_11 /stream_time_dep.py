import functions as f
import numpy as np
import matplotlib.pyplot as plt

# Plots a Time-Dependent Fourier Trajectory in a Stream Function.

# Initial condition
x0 = 0
y0 = 0
t0 = 0
dt = 0.1

# Iteration values.
K = 10
L = 10
N = 500
T = N*dt
omega = 0# 2*np.pi*50/T

# Other variables.
mu = 0
sigma = 0.2
p = 1
M = 10
epsilon = 0.1


plot = True
saving = True
perturbation = False


# COmputing alpha and beta coefficients for each pair of k,l.
coefficients = f.coefficient(K, L, mu, sigma, p)

# Computing the stream function for the background of plot.
heatmap, x_grid, y_grid = f.psi(K, L, coefficients)

# Computing the forward trajectory.
forward_trajectory = f.trajectory(
                        x0, y0, t0, dt, K, L, coefficients, N, saving, omega
                            )
if saving:
    x_values, y_values, t_values = forward_trajectory

# Computing the backwards trajectory.
if perturbation:
    pert_back_x, pert_back_y = f.perturbation(
        M, x_values, y_values, epsilon, f.trajectory, t_values,
        dt, K, L, coefficients, N, omega, saving)

backwards_trajectory = f.trajectory(
                        x_values[-1], y_values[-1], t_values[-1], -dt, K, L,
                        coefficients, N, saving, omega
                        )
if saving:
    x_back, y_back, t_back = backwards_trajectory

# Plotting the stream function as a heatmap.
plt.figure(figsize=(8, 6))
plt.imshow(
    heatmap,
    # Edge are the extent of the map
    extent=(x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]),
    # (0,0) is bottom left
    origin="lower",
    # Square pixel
    aspect="equal",
    # Colour scheme other option viridis
    cmap="plasma"
)

# Plotting the trajectories on top.
if plot and saving:
    # Forward values (possible to change point size s=2).
    plt.scatter(x_values, y_values, label="forward", s=10, color="white")
    # Backward values.
    plt.scatter(x_back, y_back, label="backward", s=10, color="black")
    if perturbation:
        plt.scatter(pert_back_x, pert_back_y, label="backward perturbed", s=10, color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    # Initial point.
    plt.scatter(x_values[0], y_values[0], label='x0')
    # Final point of forward/ first of backward.
    plt.scatter(x_values[-1], y_values[-1], label='xN')
    # FInal point of baclwards trajectory.
    plt.scatter(x_back[-1], y_back[-1], label='return point')
    plt.title("Time-Dependent Fourier Trajectory in a Stream Function.")
    # Scale for heatmap.
    plt.colorbar(label="Stream values")
    # Ensuring the legend is not on the plot.
    plt.legend(loc="center left", bbox_to_anchor=(1.25, 0.5))
    plt.tight_layout()
    plt.show()
