import numpy as np


def forward_euler(x: float, y: float, dt: float):
    """
    Forward Euler approximation function

    Args:
        x (float): The first x value
        y (float): The first y value
        dt (float): The time step

    Returns:
        X (float): The next x value
        Y (float): The next y value
    """
    X = x - dt*np.cos(y)
    Y = y + dt*np.cos(x)
    return X, Y


def symplectic_euler(x, y, dt):
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
    X = x - dt*np.cos(y)
    Y = y + dt*np.cos(X)
    return X, Y


def adjoint_symplectic_euler(x, y, dt):
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
    Y = y + dt*np.cos(x)
    X = x - dt*np.cos(Y)
    return X, Y


def composition(x, y, dt):
    """
    Composition of the two symplectic Euler functions

    Args:
        x (float): The first x value
        y (float): The first y value
        dt (float): The time step

    Returns:
        X (float): The next x value
        Y (float): The next y value
    """
    Y = y + dt/2*np.cos(x)
    X = x - dt*np.cos(Y)
    Y = Y + dt/2*np.cos(X)
    return X, Y
