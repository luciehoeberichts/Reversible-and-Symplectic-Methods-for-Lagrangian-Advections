import numpy as np
"""
This file contains the functions to compute
Symplectic Euler, Adjoint Symplectic Euler
and Symmetric Symplectic.
They will be used to compute the perturbation heatmaps.
"""


# SYMPLECTIC EULER
def symplectic_euler(x0, y0, dt, N):
    """
    Symplectic Euler approximation (conserves energy)

    Args:
        x (float): The first x value
        y (float): The first y value
        dt (float): The time step

    Returns:
        X (float): The next x value
        Y (float): The next y value
    """
    x = x0
    y = y0
    for _ in range(N):
        x = x - dt*np.cos(y)
        y = y + dt*np.cos(x)
    return x, y


# ADJOINT SYMPLECTIC EULER
def adjoint_symplectic_euler(x0, y0, dt, N):
    """
    Adjoint Symplectic Euler approximation

    Args:
        x (float): The first x value
        y (float): The first y value
        dt (float): The time step

    Returns:
        X (float): The next x value
        Y (float): The next y value
    """
    x = x0
    y = y0
    for _ in range(N):
        y = y + dt*np.cos(x)
        x = x - dt*np.cos(y)
    return x, y


def composition_euler(x0, y0, dt, N):
    """
    Adjoint Symplectic Euler approximation

    Args:
        x (float): The first x value
        y (float): The first y value
        dt (float): The time step

    Returns:
        X (float): The next x value
        Y (float): The next y value
    """
    x = x0
    y = y0
    for _ in range(N):
        y = y + dt/2*np.cos(x)
        x = x - dt*np.cos(y)
        y = y + dt/2*np.cos(x)
    return x, y
