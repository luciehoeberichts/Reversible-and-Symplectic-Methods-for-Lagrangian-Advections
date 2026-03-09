import numpy as np
import matplotlib.pyplot as plt

# Importing function options
import approximation_functions as app
import forward_backward_functions as fb
# # from approximation_functions import forward_euler
# from approximation_functions import symplectic_euler
# from approximation_functions import adjoint_symplectic_euler
# # from approximation_functions import composition

# # Importing simulation functions
# from forward_backward_functions import forward_function
# from forward_backward_functions import backward_function

"""
Plots the difference between forward and backwards arrays.
"""

x0 = 2.2
y0 = -1.0
dt = 0.001
T = 50
forward_choice = app.symplectic_euler
forward_array, N, xN, yN = fb.forward_function(x0, y0, dt, T, forward_choice)

backward_choice = app.adjoint_symplectic_euler
dt = -dt
backward_array = fb.backward_function(xN, yN, dt, T, backward_choice)

# print(forward_array)
# # print(backward_array)

difference_array = forward_array - backward_array

# print(difference_array)

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
