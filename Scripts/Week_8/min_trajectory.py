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
i_min, j_min = np.unravel_index(np.argmin(heatmap), heatmap.shape)

plt.figure(figsize=(6, 6))

# Finds the point from which it came from in the grid
x0_min = X[i_min, j_min]
y0_min = Y[i_min, j_min]

x_trajectory_min = [x0_min]
y_trajectory_min = [y0_min]

# Forward image
x_min, y_min = x0_min, y0_min
for _ in range(N):
    # Applying forward and storing each point
    x_min, y_min = movement(x_min, y_min, dt, 1, app.composition)

    x_trajectory_min.append(x_min)
    y_trajectory_min.append(y_min)

plt.plot(x_trajectory_min, y_trajectory_min, label='Forward Trajectory')

# Create lists to store all pertubation points and final return points
x1_tildes_min = []
y1_tildes_min = []

for m in range(M):
    # Apply pertubation
    theta = 2*np.pi*m/M
    x1_tilde_min = x_min + epsilon*np.cos(theta)
    y1_tilde_min = y_min + epsilon*np.sin(theta)
    x1_tildes_min.append(x1_tilde_min)
    y1_tildes_min.append(y1_tilde_min)

    x_back_min = [x1_tilde_min]
    y_back_min = [y1_tilde_min]

    # Apply backward function
    x_final_min, y_final_min = x1_tilde_min, y1_tilde_min

    for _ in range(N):
        x_final_min, y_final_min = movement(
            x_final_min, y_final_min, -dt, 1, app.composition)
        x_back_min.append(x_final_min)
        y_back_min.append(y_final_min)

    plt.plot(x_back_min, y_back_min)
    plt.scatter(x_back_min[-1], y_back_min[-1], marker='x')
plt.scatter(x0_min, y0_min, label='Initial point')
plt.scatter(x_min, y_min, label='Forward Image')
plt.scatter(x1_tildes_min, y1_tildes_min, label='Perturbed points')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory of the Minimum Return Distance Point")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
