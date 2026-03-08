import numpy as np
import matplotlib.pyplot as plt
# from approximation_functions import forward_euler
from approximation_functions import symplectic_euler
from approximation_functions import adjoint_symplectic_euler
# from approximation_functions import composition
"""
Calculates the difference in values going forward (dt) and backward (-dt)

Args:
    x0 (float): The initial x value
    y0 (float): The initial y value
    dt (float): The time step
    T (float): Total time of simulation
    forward_function (function): choice of forward function
    backward_function (function):  choice of backward function

Returns:
    forward_array (np.array): 2*(N+1) array of values for forward function
    backward_array (np.array): 2*(N+1) array of values for forward function
    delta_array (np.array): 2*(N+1) array of delta difference between the two

The function option are.
    - Forward Euler
    - Symplectic Euler
    - Adjoint Symplectic Euler
    - Composition of the two symplectic Euler functions

"""



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


x0 = 2.2
y0 = -1.0
dt = 0.001
T = 50
forward_choice = symplectic_euler
forward_array, N, xN, yN = forward_function(x0, y0, dt, T, forward_choice)

backward_choice = adjoint_symplectic_euler
dt = -dt
backward_array = backward_function(xN, yN, dt, T, backward_choice)

print(forward_array)
# print(backward_array)

difference_array = forward_array - backward_array
print(difference_array)

time = np.linspace(0, T, N+1)

plt.plot(time, difference_array[0, :], label="Δx")
plt.plot(time, difference_array[1, :], label="Δy")

plt.xlabel("time")
plt.ylabel("difference")
plt.title("Forward vs Backward Difference")
plt.legend()
plt.show()

# plt.plot(time, forward_array[0, :])
# plt.plot(time, backward_array[0, :])
# plt.show()
