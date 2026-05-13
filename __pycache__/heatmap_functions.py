import numpy as np
"""
This file contains the functions to compute Symplectic Euler,
Adjoint Symplectic Euler and Störmer-Verlet.
They will be used to compute displacement maps.
"""


# ------------------------------------MORE FLOWS------------------------------
def partials(flow):
    """
    Returns the functions for partial derivatives of a given Hamiltonian.

    Args:
        flow (str): choice of Hamiltonian

    Returns:
        partial_x (func): the function that will return the partial
        deriviative in regards to x of the given Hamiltonian.
        partial_y (func): the function that will return the partial
        deriviative in regards to x of the given Hamiltonian.
    """
    if flow == "sine":
        # H(x, y) = sin(x) + sin(y)
        def partial_x(x):
            return np.cos(x)

        def partial_y(y):
            return np.cos(y)

    elif flow == "harmonic_oscillator":
        # H(x, y) = x**2 + y**2
        def partial_x(x):
            return 2*x

        def partial_y(y):
            return 2*y

    elif flow == "pendulum":
        # H(x, y) = 1/2*y - cos(x)
        def partial_x(x):
            return np.sin(x)

        def partial_y(y):
            return y
    return partial_x, partial_y


# --------------------------------FORWARD EULER ------------------------------
def forward_euler(x0, y0, dt, N, partial_x, partial_y):
    x = x0
    y = y0
    for _ in range(N):
        x_temp = x
        x = x - dt*partial_y(y)
        y = y - dt*partial_x(x_temp)

    return x, y


# ------------------------------------SYMPLECTIC EULER------------------------
def symplectic_euler(x0, y0, dt, N, partial_x, partial_y):
    """
    Symplectic Euler method (conserves energy).

    Args:
        x0 (float): The first x value
        y0 (float): The first y value
        dt (float): The time step
        N (float): The simulation time
        partial_x (func): the function that returns the partial x derivative.
        partial_y (func): the function that returns the partial y derivative.

    Returns:
        x (float): The final x value
        y (float): The final y value
    """
    x = x0
    y = y0
    for _ in range(N):
        x = x - dt*partial_y(y)
        y = y + dt*partial_x(x)
    return x, y


# ------------------------------------ADJOINT SYMPLECTIC EULER---------------
def adjoint_symplectic_euler(x0, y0, dt, N, partial_x, partial_y):
    """
    Adjoint Symplectic Euler method.

    Args:
        x0 (float): The first x value
        y0 (float): The first y value
        dt (float): The time step
        N (float): The simulation time
        partial_x (func): the function that returns the partial x derivative.
        partial_y (func): the function that returns the partial y derivative.

    Returns:
        x (float): The final x value
        y (float): The final y value
    """
    x = x0
    y = y0
    for _ in range(N):
        y = y + dt*partial_x(x)
        x = x - dt*partial_y(y)
    return x, y


# ------------------------------------STÖRMER-VERLET------------------------------
def composition_euler(x0, y0, dt, N, partial_x, partial_y):
    """
    Self-Adjoint Störmer-Verlet method.

    Args:
        x0 (float): The first x value
        y0 (float): The first y value
        dt (float): The time step
        N (float): The simulation time
        partial_x (func): the function that returns the partial x derivative.
        partial_y (func): the function that returns the partial y derivative.

    Returns:
        x (float): The final x value
        y (float): The final y value
    """
    x = x0
    y = y0
    for _ in range(N):
        y = y + dt/2*partial_x(x)
        x = x - dt*partial_y(y)
        y = y + dt/2*partial_x(x)
    return x, y
