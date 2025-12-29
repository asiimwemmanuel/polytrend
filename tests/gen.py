import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate x values
x_val = np.linspace(-5, 5, 20)  # 20 points from -5 to 5

# Define the true polynomial function (e.g., cubic)
def true_poly(x):
    return 2*x**3 - 3*x**2 + x + 5

# Simulate measurement errors
y_err = np.random.uniform(1.0, 3.0, size=x_val.shape)  # random error between 1 and 3

# Generate noisy y values
y_val = true_poly(x_val) + np.random.normal(0, y_err)

# Print arrays
print("x_val:", x_val)
print("y_val:", y_val)
print("y_err:", y_err)

