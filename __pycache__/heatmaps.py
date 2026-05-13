import numpy as np
import heatmap_functions as f
import matplotlib.pyplot as plt

"""
------------flow options:
sine: H(x, y) = sin(x) + sin(y)
harmonic_oscillator: H(x,y) = x^2 + y^2
pendulum: H(x, y) = y/2 - cos(x)

------------perturbation options:
False: will return the displacement map of the returning points after
forward and backward trajectories.
True: will return the displacement map of the ensemble return after
a perturbation was applied.

------------max_traj options:
False: returns nothing.
True: returns the trajectory of the point that returned the largest
distance from intial condition.

------------forward_choice/backward_choic options:
f.symplectic_euler: uses the symplectic Euler method.
f.adjoint_symplectic_euler: uses the adjoint to the symplectic Euler method.
f.composition: uses the self-adjoint Störmer-Verlet method.
"""

# ----------------------------------OPTIONS----------------------------------
flow = "pendulum"  # Choice of Hamiltonian.
forward_choice = f.composition_euler  # Choice of forward method.
backward_choice = f.composition_euler  # Choice of bakward method.
# Retrieves the corresponding partial derivatives.
partial_x, partial_y = f.partials(flow)
perturbation = True  # Choice if perturbation before return.
max_traj = False  # Choice if plot the the trajectory of largest distance.

dt = 0.01  # Time step size.
N = 1000  # Number of steps.
if perturbation:
    M = 10  # Number of points on perturbation circle.
    epsilon = 0.0001  # perturbation size.

# ----------------------------------INITIAL CONDITIONS------------------------
# Making two arrays of 100x100 equally distanced points that will be grid axes.
x_values = np.linspace(-np.pi, np.pi, 100)
y_values = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x_values, y_values)  # Creating the grid from the axes.
X1 = np.empty_like(X)  # Setting the grid for the x1 values.
Y1 = np.empty_like(Y)
heatmap = np.zeros_like(X, dtype=float)  # Creating the grid of success rate.

# ----------------------------------SIMULATION--------------------------------
for i in range(len(x_values)):  # Looping over each point in the grid.
    for j in range(len(y_values)):
        x0 = X[i, j]  # Setting the point as initial condition for function.
        y0 = Y[i, j]

        # Applying the forward function to x0, y0 and getting x1, y1.
        x1, y1 = forward_choice(x0, y0, dt, N, partial_x, partial_y)

        X1[i, j] = x1  # Adding the point to the X1 grid.
        Y1[i, j] = y1

        if perturbation:  # --------PERTURBATION------------------------------
            sum_distance = 0

            for m in range(M):  # Creating M number of points.
                theta = 2*np.pi*m/M  # Equally spacing each point on circle.
                x1_tilde = x1 + epsilon*np.cos(theta)
                y1_tilde = y1 + epsilon*np.sin(theta)

                # Applying backward approximation.
                x_final, y_final = backward_choice(
                        x1_tilde, y1_tilde, -dt, N, partial_x, partial_y)

                # Calculating the distance between inital point and end point.
                distance = np.sqrt((x_final-x0)**2 + (y_final-y0)**2)
                sum_distance += distance
            # Taking the mean of the M distances per point.
            # and setting it on heatmap.
            heatmap[i, j] = sum_distance/M/epsilon

        else:  # -------------------NO PERTURBATION---------------------------
            # Apply backward function from arrival point.
            x_final, y_final = backward_choice(
                x1, y1, -dt, N, partial_x, partial_y)
            # Calculating distance and setting it to heatmap.
            distance = np.sqrt((x_final - x0)**2+(y_final - y0)**2)
            heatmap[i, j] = distance
# ----------------------------------MAXIMUM TRAJECTORY------------------------
if max_traj:
    plt.figure(figsize=(6, 6))
    # Finds the coordinate of the point with the largest
    # value/distance on heatmap grid.
    i_max, j_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    x0_max = X[i_max, j_max]  # Setting that coordinate as the initial.
    y0_max = Y[i_max, j_max]  # condition from the X/Y grid.

    x_trajectory_max = [x0_max]  # Creating lists for the forward trajectory.
    y_trajectory_max = [y0_max]

    # Forward image
    x_max, y_max = x0_max, y0_max
    for _ in range(N):
        x_max, y_max = forward_choice(  # Applying forward function.
            x_max, y_max, dt, 1, partial_x, partial_y)

        x_trajectory_max.append(x_max)  # Storing each point.
        y_trajectory_max.append(y_max)
    # Plotting the trajectory.
    plt.plot(x_trajectory_max, y_trajectory_max, label='Forward Trajectory')

    x1_tildes_max = []  # Creating lists to store all perturbation points.
    y1_tildes_max = []

    for m in range(M):  # Applying perturbation.
        theta = 2*np.pi*m/M
        x1_tilde_max = x_max + epsilon*np.cos(theta)
        y1_tilde_max = y_max + epsilon*np.sin(theta)
        x1_tildes_max.append(x1_tilde_max)
        y1_tildes_max.append(y1_tilde_max)

        x_back_max = [x1_tilde_max]  # Lists to store the backward trajectory.
        y_back_max = [y1_tilde_max]

        x_final_max, y_final_max = x1_tilde_max, y1_tilde_max
        for _ in range(N):  # Applying backward function.
            x_final_max, y_final_max = backward_choice(
                    x_final_max, y_final_max, -dt, 1, partial_x, partial_y)

            x_back_max.append(x_final_max)  # Storing each point.
            y_back_max.append(y_final_max)
        # Plotting each backward trajectory for each
        # m and x-ing each final point.
        plt.plot(x_back_max, y_back_max)
        plt.scatter(x_back_max[-1], y_back_max[-1], marker='x')
    # Marking important points.
    plt.scatter(x0_max, y0_max, label='Initial point')
    plt.scatter(x_max, y_max, label='Forward Image')
    plt.scatter(x1_tildes_max, y1_tildes_max, label='Perturbed points')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.show()

plt.figure(figsize=(8, 6))  # Plotting heatmap.
plt.imshow(
    np.log(heatmap),
    # Edge are the extent of the map
    extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
    origin="lower",  # (0,0) is bottom left
    aspect="equal",  # Square pixel
    cmap="coolwarm"  # Colour scheme other option viridis
)
cbar = plt.colorbar()
cbar.set_label("Distance", fontsize=20)
cbar.ax.tick_params(labelsize=14)

plt.xlabel("$x_0$", fontsize=16)
plt.ylabel("$y_0$", fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()
