import numpy as np
import final_functions as f
import matplotlib.pyplot as plt

"""
Plots the distance after coming back using a heatmap
Averages the distance per point because we are ading M perturbation points
"""
np.random.seed(1)

forward_choice = f.composition_euler
backward_choice = f.composition_euler

dt = 0.1  # Time step
N = 100  # Number of steps
perturbation = True
max_traj = False

if perturbation:
    M = 10  # Number of points on perturbation circle
    epsilon = 0.001  # perturbation size

# Making two arrays of 50 equally distanced points that will be grid axes.
x_values = np.linspace(-np.pi, np.pi, 100)
y_values = np.linspace(-np.pi, np.pi, 100)

# Creating the grid from the axes.
X, Y = np.meshgrid(x_values, y_values)

# Setting the grid for the x1 values
X1 = np.empty_like(X)
Y1 = np.empty_like(Y)

# creating the grid of success rate
heatmap = np.zeros_like(X, dtype=float)

# Looping over each point in the grid
for i in range(len(x_values)):
    for j in range(len(y_values)):
        x0 = X[i, j]
        y0 = Y[i, j]

        # Applying the forward function to x0, y0 and getting x1, y1
        x1, y1 = forward_choice(x0, y0, dt, N)

        X1[i, j] = x1
        Y1[i, j] = y1

        # Adding the perturbation
        if perturbation:
            sum_distance = 0

            for m in range(M):
                theta = 2*np.pi*m/M  # Equally spacing each point on circle
                x1_tilde = x1 + epsilon*np.cos(theta)
                y1_tilde = y1 + epsilon*np.sin(theta)

                # Applying backward approximation
                x_final, y_final = backward_choice(
                        x1_tilde, y1_tilde, -dt, N)
                # Calculating the distance between inital point an return point

                distance = np.sqrt((x_final-x0)**2 + (y_final-y0)**2)
                sum_distance += distance
            heatmap[i, j] = sum_distance/M

        else:
            x_final, y_final = backward_choice(x1, y1, -dt, N)
            distance = np.sqrt((x_final - x0)**2+(y_final - y0)**2)
            heatmap[i, j] = distance
if max_traj:
    plt.figure(figsize=(6, 6))
    i_max, j_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    # Finds the point from which it came from in the grid
    x0_max = X[i_max, j_max]
    y0_max = Y[i_max, j_max]

    # Creating lists for the forwardtrajectory
    x_trajectory_max = [x0_max]
    y_trajectory_max = [y0_max]

    # Forward image
    x_max, y_max = x0_max, y0_max
    for _ in range(N):
        # Applying forward and storing each point
        x_max, y_max = forward_choice(x_max, y_max, dt, 1)

        x_trajectory_max.append(x_max)
        y_trajectory_max.append(y_max)
    plt.plot(x_trajectory_max, y_trajectory_max, label='Forward Trajectory')

    # Create lists to store all perturbation points
    x1_tildes_max = []
    y1_tildes_max = []

    for m in range(M):
        # Apply perturbation
        theta = 2*np.pi*m/M
        x1_tilde_max = x_max + epsilon*np.cos(theta)
        y1_tilde_max = y_max + epsilon*np.sin(theta)
        x1_tildes_max.append(x1_tilde_max)
        y1_tildes_max.append(y1_tilde_max)

        # Creating lists to store the backward trajectories
        x_back_max = [x1_tilde_max]
        y_back_max = [y1_tilde_max]

        # Apply backward function
        x_final_max, y_final_max = x1_tilde_max, y1_tilde_max
        for _ in range(N):
            x_final_max, y_final_max = backward_choice(
                    x_final_max, y_final_max, -dt, 1)

            x_back_max.append(x_final_max)
            y_back_max.append(y_final_max)
        # Plotting each backward trajectory for each
        # m and xing each final point
        plt.plot(x_back_max, y_back_max)
        plt.scatter(x_back_max[-1], y_back_max[-1], marker='x')
    # marking important points
    plt.scatter(x0_max, y0_max, label='Initial point')
    plt.scatter(x_max, y_max, label='Forward Image')
    plt.scatter(x1_tildes_max, y1_tildes_max, label='Perturbed points')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


# PLOTS HEATMAP

plt.figure(figsize=(8, 6))
plt.imshow(
    heatmap,
    # Edge are the extent of the map
    extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
    # (0,0) is bottom left
    origin="lower",
    # Square pixel
    aspect="equal",
    # Colour scheme other option viridis
    cmap="coolwarm"
)

plt.colorbar(label="Distance ")
plt.xlabel("$x_0$")
plt.ylabel("$y_0$")
plt.show()
