import functions as f
import numpy as np


# Initial conditions
x0 = 0
y0 = 0
t0 = 0
dt = 1

# Iteration values.
K = 10
L = 10
N = 500
T = N*dt
omega = 2*np.pi*10/T

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

if perturbation:
    pert_back_x, pert_back_y = f.perturbation(
                                    M, x_values,
                                    y_values,
                                    epsilon,
                                    f.trajectory,
                                    t_values,
                                    -dt,
                                    K,
                                    L,
                                    coefficients,
                                    N,
                                    omega,
                                    saving)
