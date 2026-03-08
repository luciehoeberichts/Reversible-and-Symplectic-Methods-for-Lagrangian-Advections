import numpy as np 

def forward_function(x0, y0, dt, T, forward_choice):
    """
    Stores the values of the backward function into a dictionary.

    Args:
        x0 (float): initial condition for x coordinate
        y0 (float): initial condition for y coordinate
        dt (float): time step
        T (int): period of time for simulation
        forward_choice (function): choice of function for forward simulation

    Returns:
        forward_array (np.array[float]): the array of images
        N (int): the final step number
        x[N] (float): the final x pre-image of the function on the time period
        y[N] (float): the final y pre-image of the function on the time period
    """

    N = int(np.ceil(abs(T / dt)))

    # Creating memory space for n+1
    x = np.empty(N+1)
    y = np.empty(N+1)

    # Creating the array of images
    forward_array = np.empty((2, N+1))

    # Setting initial conditions
    x[0] = x0
    y[0] = y0
    forward_array[:, 0] = (x0, y0)

    # Applying our function recursively and storing in array left to right
    for n in range(N):
        x[n], y[n] = forward_array[:, n]
        x[n+1], y[n+1] = forward_choice(x[n], y[n], dt)
        forward_array[:, n+1] = x[n+1], y[n+1]
    return forward_array, N, x[N], y[N]

def backward_function(xN, yN, dt, T, backward_choice):
    """
    Stores the values of the backward function into a dictionary.

    Args:
        xN (float): the x coordinate of final point of forward_function
        yN (float): the y coordinate of final point of forward_function
        dt (float): time step
        T (int): period of time for simulation
        forward_choice (function): choice of function for forward simulation

    Returns:
        x (float): the final x image of the function of the time period
        y (float): the final y image of the function of the time period
    """

    N = int(np.ceil(abs(T / dt)))

    # Creating memory space for n+1
    x = np.empty(N+1)
    y = np.empty(N+1)

    # Creating the array of images
    backward_array = np.empty((2, N+1))

    # Setting initial conditions
    x[N] = xN
    y[N] = yN
    backward_array[:, N] = (xN, yN)

    # Applying our function recursively and storing in array from right to left
    for n in range(N, 0, -1):
        x[n], y[n] = backward_array[:, n]
        x[n-1], y[n-1] = backward_choice(x[n], y[n], dt)
        backward_array[:, n-1] = x[n-1], y[n-1]
    return backward_array