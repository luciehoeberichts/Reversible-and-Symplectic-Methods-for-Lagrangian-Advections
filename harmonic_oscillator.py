import numpy as np
import matplotlib.pyplot as plt


def H(x, y):
    # Hamiltonian of Harmonic Oscillator
    return x**2 + y**2


def euler(x, y, dt):
    # One step Euler method
    X = x + dt*y
    Y = y - dt*X
    return X, Y


def euler_trajectory(x0, y0, dt, T):

    N = int(np.ceil(abs(T / dt)))

    # Creating memory space for n+1
    x = np.empty(N+1)
    y = np.empty(N+1)

    # Setting initial conditions
    x[0] = x0
    y[0] = y0

    for n in range(N):
        x[n+1], y[n+1] = euler(x[n], y[n], dt)

    return x, y


def plotting(x0=2.2, y0=-1.0, dt=0.05, T=50.0,  use_scatter=True):
    # this function plots the graph of Euler
    # approximation with given parameters
    x_path, y_path = euler_trajectory(x0, y0, dt, T)

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
    plt.title(f"x^2 + y^2 vs Euler approximation (x0={x0}, y0={y0}, dt={dt}, T={T})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plotting(x0=2.2, y0=-1.0, dt=-0.005, T=80.0, use_scatter=True)
