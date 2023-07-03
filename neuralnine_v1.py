import numpy as np
import matplotlib.pyplot as plt

# Generate random data
x = np.random.rand(100, 1)
y = 4 + 2 * x + np.random.randn(100, 1)

# Add polynomial features to x
degree = 1 # Set the degree of the polynomial features
x_poly = np.concatenate((x, x ** 2), axis=1)

# Solve for the polynomial coefficients using normal equations
coefficients = np.linalg.inv(x_poly.T @ x_poly) @ x_poly.T @ y

# Generate predictions
x_vals = np.linspace(0, 1, 100).reshape(-1, 1)
x_vals_poly = np.concatenate((x_vals, x_vals ** 2), axis=1)
y_vals = x_vals_poly @ coefficients

# Plot the results
plt.scatter(x, y)
plt.plot(x_vals, y_vals, color='r')
plt.show()
