import numpy as np
import approximation_functions as app
from pertubation_functions import movement
import matplotlib.pyplot as plt

# Plots the trajectory of the largest and smallest return distances
# Is identical to pertubation_map.py until line 56

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

# Finding index of the point that gives the biggest pertubation
# argmax flattens the indices of the array and then unravel puts it back
# Finds the largest point of pertubation
i_max, j_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
i_min, j_min = np.unravel_index(np.argmin(heatmap), heatmap.shape)

# Finds the point from which it came from in the grid
x0_max = X[i_max, j_max]
y0_max = Y[i_max, j_max]
x0_min = X[i_min, j_min]
y0_min = Y[i_min, j_min]

# Forward image
x1_max, y1_max = movement(x0_max, y0_max, dt, N, app.composition)
x1_min, y1_min = movement(x0_min, y0_min, dt, N, app.composition)

# Create lists to store all pertubation points and final return points
x1_tildes_max = []
y1_tildes_max = []
x_finals_max = []
y_finals_max = []

x1_tildes_min = []
y1_tildes_min = []
x_finals_min = []
y_finals_min = []

for m in range(M):
    # Apply pertubation
    theta = 2*np.pi*m/M
    x1_tilde_max = x1_max + epsilon*np.cos(theta)
    y1_tilde_max = y1_max + epsilon*np.sin(theta)
    x1_tildes_max.append(x1_tilde_max)
    y1_tildes_max.append(y1_tilde_max)

    x1_tilde_min = x1_min + epsilon*np.cos(theta)
    y1_tilde_min = y1_min + epsilon*np.sin(theta)
    x1_tildes_min.append(x1_tilde_min)
    y1_tildes_min.append(y1_tilde_min)

    # Apply backward function
    x_final_max, y_final_max = movement(
        x1_tilde_max, y1_tilde_max, -dt, N, app.composition)
    x_finals_max.append(x_final_max)
    y_finals_max.append(y_final_max)

    x_final_min, y_final_min = movement(
        x1_tilde_min, y1_tilde_min, -dt, N, app.composition)
    x_finals_min.append(x_final_min)
    y_finals_min.append(y_final_min)

# # Plotting MAXIMUM trajectory
# plt.figure(figsize=(8, 8))

# # Initial point
# plt.scatter(x0_max, y0_max, label='Initial point')

# # Forward imag before pertubation
# plt.scatter(x1_max, y1_max, label='Forward Image')

# # Pertubation points
# plt.scatter(x1_tildes_max, y1_tildes_max, label='Pertubation Points')

# # Return points
# plt.scatter(x_finals_max, y_finals_max, label='Final Return Points')

# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Trajectory of the Maximum Return Distance Point")
# plt.legend()
# plt.axis("equal")
# plt.grid(True)
# plt.show()

# Plotting MINIMUM trajectory
plt.figure(figsize=(8, 8))

# Initial point
plt.scatter(x0_min, y0_min, label='Initial point')

# Forward imag before pertubation
plt.scatter(x1_min, y1_min, label='Forward Image')

# Pertubation points
plt.scatter(x1_tildes_min, y1_tildes_min, label='Pertubation Points')

# Return points
plt.scatter(x_finals_min, y_finals_min, label='Final Return Points')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory of the Minimum Return Distance Point")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
