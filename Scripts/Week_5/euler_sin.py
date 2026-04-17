import numpy as np
import matplotlib.pyplot as plt

# I want to plot the Euler approximation of the function sin(x) + sin(y)
# This uses two recusrive functions:
# X_{n+1} = X_n - dt*cos(y)
# Y_{n+1} = Y_n + dt*cos(x)
# This should show that Euler's approximation deviates from the solution
# with a lot of deviation in the intersection parts of the solution



def H(x, y):
    """
    Args: 
        two points
    Returns: 
        float: the hamiltonian image 
    
    """
    # Real Hamiltonian function to plot against
    return np.sin(x) + np.sin(y)


def euler_function(x, y, dt):
    # Euler approximation function
    X = x - dt*np.cos(y)
    Y = y + dt*np.cos(x)
    return X, Y


def euler_trajectory(x0, y0, dt, T, wrap=True):
    # This traces the Euler path by recursively defining it
    # and applying the function we defined
    # x0 y0 = initial conditions
    # dt = the time step
    # N = number of steps for Euler
    # T = the total time simulation
    # is for the graph to stay on the scale

    # Ensure that the programme can run witha negative dt
    N = int(np.ceil(abs(T / dt)))

    # Creating memory space for n+1
    x = np.empty(N+1)
    y = np.empty(N+1)

    # Setting initial conditions
    x[0] = x0
    y[0] = y0

    # Applying our function ecursively
    for n in range(N):
        x[n+1], y[n+1] = euler_function(x[n], y[n], dt)

        if wrap:
            # these are from gpt for the plotting on target
            x[n+1] = (x[n+1] + np.pi) % (2*np.pi) - np.pi
            y[n+1] = (y[n+1] + np.pi) % (2*np.pi) - np.pi

    return x, y


def plotting(x0=2.2, y0=-1.0, dt=0.05, T=50.0, wrap=True, use_scatter=True):
    # this function plots the graph of Euler
    # approximation with given parameters
    x_path, y_path = euler_trajectory(x0, y0, dt, T, wrap=wrap)

    # creating the coordinate grid for H
    xg = np.linspace(-np.pi, np.pi, 400)
    yg = np.linspace(-np.pi, np.pi, 400)
    X, Y = np.meshgrid(xg, yg)
    Z = H(X, Y)

    # Plotting H
    plt.figure(figsize=(8, 8))
    plt.contour(X, Y, Z, levels=25, colors="black", alpha=0.6)

    # Pollting Euler
    if use_scatter:
        # Scatter avoids fake straight lines caused by wrapping jumps
        plt.scatter(x_path, y_path, s=2, color="red", label="Euler")
    else:
        plt.plot(x_path, y_path, color="red", linewidth=1.2, label="Euler")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"sin(x) + sin(y) vs Euler approximation (x0={x0}, y0={y0}, dt={dt}, T={T})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plotting(x0=2.2, y0=-.0, dt=0.2, T=1000.0, wrap=True, use_scatter=True)
