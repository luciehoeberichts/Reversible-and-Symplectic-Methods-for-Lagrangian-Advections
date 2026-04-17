
def movement(x, y, dt, N, choice):
    """
    Gives the final point that the function is applied to.

    Args:
        x0 (float): initial condition for x coordinate
        y0 (float): initial condition for y coordinate
        dt (float): time step
        T (int): period of time for simulation
        choice (function): choice of function for simulation

    Returns:
        x (float): the final x
        y (float): the final y
    """

    # Applying our function recursively and storing in array left to right
    for _ in range(N):
        x, y = choice(x, y, dt)
    return x, y
