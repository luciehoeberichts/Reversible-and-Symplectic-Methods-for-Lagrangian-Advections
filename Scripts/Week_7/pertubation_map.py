import numpy as np
import approximation_functions as app
from pertubation_functions import movement
import matplotlib.pyplot as plt

# Plots the distance after coming back using a heatmap
# Averages the distance per point because we are ading M pertubation points

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

dt = 0.1  # Time step
N = 100  # Number of steps
M = 10  # Number of points on pertubation circle
epsilon = 0.1  # Pertubation size

# Looping over each point in the grid
for i in range(len(x_values)):
    for j in range(len(y_values)):
        sum_distance = 0
        x0 = X[i, j]
        y0 = Y[i, j]

        # Applying the forward function to x0, y0 and getting x1, y1
        x1, y1 = movement(x0, y0, dt, N, app.composition)

        X1[i, j] = x1
        Y1[i, j] = y1

        # Adding the pertubation
        for m in range(M):
            theta = 2*np.pi*m/M  # Equally spacing each point on circle
            x1_tilde = x1 + epsilon*np.cos(theta)
            y1_tilde = y1 + epsilon*np.sin(theta)

            # Applying backward approximation
            x_final, y_final = movement(
                x1_tilde, y1_tilde, -dt, N, app.composition)

            # Calculating the distance between inital point an return point
            distance = np.sqrt((x_final-x0)**2 + (y_final-y0)**2)
            sum_distance += distance

        heatmap[i, j] = sum_distance/M

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
    # Darkest colour
    # vmin=0,
    # Brightest colour
    # vmax=100,
    # Colour scheme other option viridis
    cmap="coolwarm"
)

plt.colorbar(label="Distance ")
plt.xlabel("$x_0$")
plt.ylabel("$y_0$")
plt.title("Distance From Initial Grid Point")
plt.show()
