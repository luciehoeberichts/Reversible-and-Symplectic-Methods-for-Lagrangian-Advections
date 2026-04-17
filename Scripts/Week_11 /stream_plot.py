import numpy as np
import matplotlib.pyplot as plt


def coefficient(K, L, mu, sigma, p):
    """
    Defines fixed coefficients for each combination of k,l.

    Args:
        K (int): value to sum over.
        L (int): value to sum over.
        mu (float): mean of normal distribution.
        sigma (float): standard deviation.

    Returns:
        coefficients (dict[tuple]): keys are k, l pairs and values are
        the corresponding alpha, beta coefficients.
    """
    coefficients = {}
    for k in range(1, K+1):
        for l in range(1, L+1):
            alpha = np.random.normal(mu, sigma)/(k**2+l**2)**p
            beta = np.random.normal(mu, sigma)/(k**2+l**2)**p
            coefficients[(k, l)] = (alpha, beta)
    return coefficients


# Making two arrays of 50 equally distanced points that will be grid axes.


def psi(K, L, coefficients):
    """
    Creates a heatmap of the stream function

    Args:
        K (int): value to sum over.
        L (int): value to sum over.
        coefficients (dict): dictionary computed by coefficient function.

    Returns:
        heatmap (np.darray): uses i, j coordinates and matches the value with
        the image of the stream function.
    """
    x_values = np.linspace(0, 100, 100)
    y_values = np.linspace(0, 100, 100)

    # Creating the grid from the axes.
    X, Y = np.meshgrid(x_values, y_values)
    heatmap = np.zeros((len(y_values), len(x_values)), dtype=float)

    delta_x = 2*np.pi/100
    delta_y = 2*np.pi/100

    for i in range(len(x_values)):
        for j in range(len(y_values)):
            xxi = i*delta_x
            yyj = j*delta_y

            sum = 0
            for k in range(1, K+1):
                for l in range(1, L+1):
                    alpha, beta = coefficients[(k, l)]
                    sum += alpha*np.cos(k*xxi+l*yyj) + beta*np.sin(k*xxi+l*yyj)
            heatmap[(i, j)] = sum
    return heatmap, x_values, y_values


K = 10
L = 10
mu = 0
sigma = 0.2
p = 1

coefficients = coefficient(K, L, mu, sigma, p)
heatmap, x_values, y_values = psi(K, L, coefficients)

plt.figure(figsize=(8, 6))
plt.imshow(
    heatmap,
    # Edge are the extent of the map
    extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
    # (0,0) is bottom left
    origin="lower",
    # Square pixel
    aspect="equal",
    # Colour scheme other option viridis
    cmap="plasma"
)
plt.show()
