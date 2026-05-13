import fourier_functions as f
import numpy as np
import matplotlib.pyplot as plt

# Plots a Time-Dependent Fourier Trajectory in a Stream Function.
np.random.seed(7777)
# Initial condition
x0 = 1
y0 = 1
t0 = 0
dt = 0.1

# Iteration values.
K = 10
L = 10
N = 1000
T = N*dt
omega = 2*np.pi*50/T  # 0

# Other variables.
mu = 0
sigma = 0.2
p = 1
M = 10
epsilon = 0.1


plot = True
saving = True
perturbation = True


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
    pert_back_x, pert_back_y, back_x, back_y = f.perturbation(
        M, x_values, y_values, epsilon, f.trajectory, t_values,
        dt, K, L, coefficients, N, omega, saving)

backwards_trajectory = f.trajectory(
                        x_values[-1], y_values[-1], t_values[-1], -dt, K, L,
                        coefficients, N, saving, omega
                        )
if saving:
    x_back, y_back, t_back = backwards_trajectory

# Plotting the stream function as a heatmap.
plt.figure(figsize=(11, 7))
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
x_values = np.array(x_values)
y_values = np.array(y_values)

x_back = np.array(x_back)
y_back = np.array(y_back)

# pert_back_x = np.array(pert_back_x)
# pert_back_y = np.array(pert_back_y)

# Plotting the trajectories on top.
if plot and saving:
    # Forward trajectory.
    plt.scatter(
        x_values,
        y_values,
        label="Forward Trajectory",
        s=10,
        color="pink")
    # Backward trajectory.
    plt.scatter(
        x_back,
        y_back,
        label="Backward Trajectory",
        s=10,
        color="black")

    # Perturbed trajectory
    if perturbation:
        plt.scatter(
            pert_back_x,
            pert_back_y,
            label="Perturbed Backward Trajectory",
            s=10,
            color="red")
        plt.scatter(
            pert_back_x[-1],
            pert_back_y[-1],
            label='Perturbed Return Point')
        plt.scatter(
            back_x,
            back_y,
            label='all Perturbed Return Point')
    plt.figure(figsize=(13, 7))

# Stream function as background.
    im = plt.imshow(
        heatmap,
        extent=(x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]),
        origin="lower",
        aspect="equal",
        cmap="plasma"
    )

if plot and saving:

    plt.scatter(
        x_values % (2*np.pi),
        y_values % (2*np.pi),
        label="Forward Trajectory",
        s=10,
        color="pink"
    )

    plt.scatter(
        x_back % (2*np.pi),
        y_back % (2*np.pi),
        label="Backward Trajectory",
        s=10,
        color="black"
    )

    if perturbation:
        plt.scatter(
            pert_back_x % (2*np.pi),
            pert_back_y % (2*np.pi),
            label="Perturbed Backward Trajectory",
            s=10,
            color="red"
        )

        plt.scatter(
            pert_back_x[-1] % (2*np.pi),
            pert_back_y[-1] % (2*np.pi),
            label="Ensemble Perturbed Return Point")
        for i in range(len(back_x)):

            if i == 0:
                label = "All Perturbed Return Points"
            else:
                label = None

            plt.scatter(
                back_x[i][-1] % (2*np.pi),
                back_y[i][-1] % (2*np.pi),
                color="blue",
                s=40,
                label=label
            )
    plt.scatter(
        x_values[0] % (2*np.pi),
        y_values[0] % (2*np.pi),
        label='Initial Point $(1,1)$'
    )

    plt.scatter(
        x_values[-1] % (2*np.pi),
        y_values[-1] % (2*np.pi),
        label='Forward Final Point'
    )

    plt.scatter(
        x_back[-1] % (2*np.pi),
        y_back[-1] % (2*np.pi),
        label='Return Point'
    )

    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    cbar = plt.colorbar(im)
    cbar.set_label("Stream Values", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    plt.subplots_adjust(right=1)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(-0.80, 0.5),
        fontsize=12
        )
    plt.show()
