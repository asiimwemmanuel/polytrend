# explain the code.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LineaRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.random.rand(100, 1) # random between 
y = 4 + 2 * x + np.random.randn(100, 1) # y = 4 + 2x + e

# using degree 1 because we know the y function
poly_features = PolynomialFeatures(degree=1, include_bias=False)

# creating polynomial features
x_poly = poly_features.fit_transform(x)

reg = LinearRegression()
reg.fit(x_poly, y)

# another bug in the first 2 parameters of linspace
x_vals = np.linspace(0, 1, 100).reshape(-1, 1)
x_vals_poly = poly_features.transform(x_vals)

y_vals = reg.predict(x_vals_poly)

plt.scatter(x, y)
plt.plot(x_vals, y_vals, color='r')
plt.show()