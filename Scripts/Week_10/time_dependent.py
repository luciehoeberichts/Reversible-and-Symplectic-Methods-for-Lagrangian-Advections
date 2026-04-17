import numpy as np
import matplotlib.pyplot as plt

# We now have a Hamiltonian that depends explicitly on time:
# H(x, y, t) and H_tilde(x, y, t, s) = H(x, y, t) + s
# This allows us to contruct a symplectic splitting method
# in which time is a constant for some steps.

saving = True
plot = True
p = 1
K = 3
L = 3


# Why are alpha and beta not dependent on t?
def alpha_func(k, l):
    """
    Alpha coefficient
    """
    mu, sigma = 2, 0.1  # mean and standard deviation
    alpha = np.random.normal(mu, sigma)/(k**2+l**2)*p
    return alpha


def beta_func(k, l):
    """
    Beta coefficient
    """
    mu, sigma = 2, 0.1
    beta = np.random.normal(mu, sigma)/(k**2+l**2)*p
    return beta


# Pre-computing the coefficients to ensure they are the same at each hal step
coefficients = {}
for k in range(1, K+1, 1):
    for l in range(1, L+1, 1):
        coefficients[(k, l)] = (alpha_func(k, l), beta_func(k, l))

def step(x, y, alpha, beta, k, l, d_tau):
    """
    Movement function
    """
    w = k*x + l*y
    dH = alpha*np.cos(w) + beta*np.sin(w)

    x_new = x + d_tau*l*dH
    y_new = y - d_tau*k*dH
    return x_new, y_new


t0: float = 10  # Initial time
t = t0  # Time that will be updated
dt = -0.1  # Time step
x0 = 0
y0 = 0
x = x0
y = y0
N = 100  # Simulation time


for _ in range(N):
    if saving:
        x_values = [x]
        y_values = [y]
        t_values = [t]

# k, l can't start a 0 because we divide alpha, beta by them
    for k in range(1, K+1, 1):
        for l in range(1, L+1, 1):
            alpha, beta = coefficients[(k, l)]
            # 1. Solving the update
            x, y = step(x, y, alpha, beta, k, l, d_tau)

    # 2. Solving the system
    t = t + dt

    # Repeating step 1 in reverse order
    for k in range(K, 0, -1):
        for l in range(L, 0, -1):
            alpha, beta = coefficients[(k, l)]
            x, y = step(x, y, alpha, beta, k, l, d_tau)

    if saving:
        x_values.append(x)
        y_values.append(y)
        t_values.append(t)

if plot:
    plt.plot(x_values, y_values)
    plt.show()
