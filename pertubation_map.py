import numpy as np
from approximation_functions import forward_euler
# from approximation_functions import symplectic_euler
# from approximation_functions import adjoint_symplectic_euler
# from approximation_functions import composition
from delta_simulator import forward_function

# Making two arrays of 50 equally distanced points that will be grid axes.
x_values = np.linspace(-5, 5, 50)
y_values = np.linspace(-5, 5, 50)

# Creating the grid from the axes.
X, Y = np.meshgrid(x_values, y_values)

for x0 in x_values:
    for y0 in y_values:
        forward_function(x0, y0, 0.1, 50, forward_euler)
