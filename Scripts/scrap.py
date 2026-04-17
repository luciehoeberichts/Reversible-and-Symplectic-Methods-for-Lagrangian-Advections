# import numpy as np

# def euler_step(x, y, dt):
#     x_new = x + dt * y
#     y_new = y - dt * x
#     return x_new, y_new

# def euler_trajectory(x0, y0, dt, T):
#     N = int(np.ceil(abs(T / dt)))
#     X = np.empty((2, N + 1), dtype=float)

#     X[0, 0] = x0
#     X[1, 0] = y0

#     for n in range(N):
#         X[0, n+1], X[1, n+1] = euler_step(X[0, n], X[1, n], dt)

#     return X

# if __name__ == "__main__":
#     X = euler_trajectory(x0=2.2, y0=-1.0, dt=0.005, T=100)

#     print("X shape:", X.shape)     # should be (2, N+1)
#     print("X =")
#     print(X)

#     # Example: print the last step (x_N, y_N)
#     print("Last state:", X[:, -1])

for i in range(5, 1):
    print(i)
# dict = {}

# dict["alpha"] = [1]

# dict["alpha"].append(2)
# print(dict)
for i in range(4, 0, -1):
    print(i)
print("hello")

K = 3
L = 3
for i in range(1, K+1):
    print(i)
print("helo")
for i in range(K, 0, -1):
    print(i)