import numpy as np


# ----------------------------------COEFFICIENT GENERATOR-----------------
def coefficient(K, L, mu, sigma, p):
    """
    Defines fixed coefficients for each combination of k,l.

    Args:
        K (int): value to sum over.
        L (int): value to sum over.
        mu (float): mean of normal distribution.
        sigma (float): standard deviation.
        p (float): degree for scaling.

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


# ----------------------------------STREAM MAP--------------------------
def psi(K, L, coefficients):
    """
    Creates a heatmap of the stream function.

    Args:
        K (int): value to sum over.
        L (int): value to sum over.
        coefficients (dict): dictionary computed by coefficient function.

    Returns:
        heatmap (np.ndarray): uses i, j coordinates and matches the value with
        the image of the stream function.
        x_values (list[int]): grid axis.
        y_values (lint[int]): grid axis.
    """
    x_values = np.linspace(0, 2*np.pi, 100)
    y_values = np.linspace(0, 2*np.pi, 100)

    # Creating the grid from the axes.
    X, Y = np.meshgrid(x_values, y_values)
    heatmap = np.zeros((len(x_values), len(y_values)), dtype=float)

    for i in range(len(x_values)):
        for j in range(len(y_values)):

            sum = 0
            for k in range(1, K+1):
                for l in range(1, L+1):
                    alpha, beta = coefficients[(k, l)]
                    sum += (
                        alpha*np.cos(k*X[i, j]+l*Y[i, j])
                        + beta*np.sin(k*X[i, j]+l*Y[i, j])
                    )
            heatmap[(i, j)] = sum
    return heatmap, x_values, y_values


# ----------------------------------HALF STEP---------------------------
def half_step(x, y, alpha, beta, k, l, d_tau, t, omega):
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

    dH = alpha*np.cos(w) - beta*np.sin(w)

    if k == 2 and l == 1:
        dH = dH*np.cos(omega*t)

    x_new = x + d_tau*l*dH
    y_new = y - d_tau*k*dH

    # WRAP
    x_new = x_new #% (2*np.pi)
    y_new = y_new #% (2*np.pi)

    return x_new, y_new


# ----------------------------------FULL STEP---------------------------
def full_step(x0, y0, t0, dt, K, L, coefficients, omega):
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
        omega (float): tidal pull.

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
            x, y = half_step(x, y, alpha, beta, k, l, d_tau, t, omega)
    t = t + dt
    for k in range(K, 0, -1):
        for l in range(L, 0, -1):
            alpha, beta = coefficients[(k, l)]
            x, y = half_step(x, y, alpha, beta, k, l, d_tau, t, omega)
    return x, y, t


# ----------------------------------TRAJECTORY-------------------------------
def trajectory(x0, y0, t0, dt, K, L, coefficients, N, saving, omega):
    """"
    Args:
        x0 (float): x coordinate of initial point.
        y0 (float): y coordinate of initial point.
        t0 (float): initial time value.
        dt (float): time step.
        K (int): value to sum over.
        L (int): value to sum over.
        coefficients (dict): dictionary computed by coefficient function.
        N (int): simulation time.
        saving (bool): determines if the trajectory will be stored.
        omega (float): tidal pull.

    Returns:
        if saving:
            x_values (list[float]): the values for the x trajectory.
            y_values (list[float]): the values for the y trajectory.
            t_values (list[float]): the values for the time trajectory.
        else:
            x (float): x coordinate after full time step.
            y (float): y coordinate after full time step.
            t (float): updated time value.
    """
    x, y, t = x0, y0, t0

    if saving:
        x_values = [x]
        y_values = [y]
        t_values = [t]

    for _ in range(N):
        x, y, t = full_step(x, y, t, dt, K, L, coefficients, omega)
        if saving:
            x_values.append(x)
            y_values.append(y)
            t_values.append(t)
    if saving:
        return x_values, y_values, t_values
    else:
        return x, y, t


# ----------------------------------PERTURBATION------------------------
def perturbation(
        M, x_values, y_values, epsilon, trajectory,
        t_values, dt, K, L, coefficients, N, omega, saving
        ):
    """"
    Adds the perturbation to the simulation.

    Args:
        M
        x_values (list[float]): the values for the x trajectory.
        y_values (list[float]): the values for the y trajectory.
        epsilon (float): perturbation size.
        trajectory (function): see line 136.
        t_values (list[float]): the values for the time trajectory.
        dt (float): time step.
        K (int): value to sum over.
        L (int): value to sum over.
        coefficients (dict): dictionary computed by coefficient function.
        N (int): simulation time.
        omega (float): tidal pull.
        saving (bool): determines if the trajectory will be stored.

    Returns:
        pert_back_x (list[float]): backwards x trajectory.
        pert_back_y (list[float]): backwards y trajectory.
    """

    back_x = []
    back_y = []

    for m in range(M):
        theta = 2*np.pi*m/M  # Equally spacing each point on circle
        x1_tilde = x_values[-1] + epsilon*np.cos(theta)
        y1_tilde = y_values[-1] + epsilon*np.sin(theta)

        x1_tilde = x1_tilde
        y1_tilde = y1_tilde

        pert_backwards_trajectory = trajectory(
                                x1_tilde, y1_tilde, t_values[-1], -dt,
                                K, L, coefficients, N, saving, omega
                                )
        if saving:
            x_back, y_back, t_back = pert_backwards_trajectory
            back_x.append(x_back)
            back_y.append(y_back)

    # Compute pointwise mean
    pert_back_x = np.mean(back_x, axis=0)
    pert_back_y = np.mean(back_y, axis=0)

    return pert_back_x, pert_back_y
