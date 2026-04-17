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


def half_step(x, y, alpha, beta, k, l, d_tau):
    """
    Movement function of one step.

    Args:
        x (float): pre-image x coordinate.
        y (float): pre-image y coordinate.
        alpha (float): pre-calculated coefficient.
        beta (float): pre-calculated coefficient.
        k, l (int): point of iteration.
        d_tau (float): half time step

    Returns:
        x_new (float): x-coordinate of image one half step later
        y_new (float): y-coordinate of image one half step later
    """
    w = k*x + l*y
    dH = alpha*np.cos(w) + beta*np.sin(w)

    x_new = x + d_tau*l*dH
    y_new = y - d_tau*k*dH
    return x_new, y_new


def full_step(x0, y0, t0, dt, K, L, coefficients):
    """
    Runs the approximation over a single time step.

    Args:
        x0 (float): x coordinate of initial point.
        y0 (float): y coordinate of initial point.
        t0 (float): initial time value.
        dt (float): time step.
        K (int): value to sum over.
        L (int): value to sum over.
        coefficients (dict): dictionary computed by coefficient function.

    Returns:
        x (float): x coordinate after full time step.
        y (float): y coordinate after full time step.
        t (float): updated time value.
        """
    x, y, t = x0, y0, t0
    d_tau = dt/2
    for k in range(1, K+1):
        for l in range(1, L+1):
            alpha, beta = coefficients[(k, l)]
            x, y = half_step(x, y, alpha, beta, k, l, d_tau)
    t = t + dt
    for k in range(K, 0, -1):
        for l in range(L, 0, -1):
            alpha, beta = coefficients[(k, l)]
            x, y = half_step(x, y, alpha, beta, k, l, d_tau)
    return x, y, t


def trajectory(x0, y0, t0, dt, K, L, coefficients, N, saving):
    x, y, t = x0, y0, t0

    if saving:
        x_values = [x]
        y_values = [y]
        t_values = [t]

    for _ in range(N):
        x, y, t = full_step(x, y, t, dt, K, L, coefficients)
        if saving:
            x_values.append(x)
            y_values.append(y)
            t_values.append(t)
    if saving:
        return x_values, y_values, t_values
    else:
        return x, y, t


# START

K = 3
L = 3
x0 = 0
y0 = 0
t0 = 0
dt = 0.1
N = 10

mu = 5
sigma = 0.2
p = 1

plot = True
saving = True

coefficients = coefficient(K, L, mu, sigma, p)
forward_trajectory = trajectory(x0, y0, t0, dt, K, L, coefficients, N, saving)
if saving:
    x_values, y_values, t_values = forward_trajectory

backwards_trajectory = trajectory(x_values[-1], y_values[-1], t_values[-1], -dt, K, L, coefficients, N, saving)
if saving:
    x_back, y_back, t_back = backwards_trajectory

print(x_back[-1] - x0)
print(y_back[-1] - y0)
print(t_back[-1] - t0)
if plot and saving:
    plt.plot(x_values, y_values)
    plt.plot(x_back, y_back)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x_values[0], y_values[0], label='x0')
    plt.scatter(x_values[-1], y_values[-1], label='xN')
    # plt.scatter(x_back[-1], y_back[-1], label='return point')
    plt.title("Trajectory")
    plt.legend()
    plt.show()
