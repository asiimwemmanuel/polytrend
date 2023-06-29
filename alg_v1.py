import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class polytrend:
	def poly(self, data):
		x = np.array([point[0] for point in data]).reshape(-1, 1)
		y = np.array([point[1] for point in data])

		best_order = None
		best_error = float('inf')
		best_model = None

		for order in range(1, 5):
			poly_features = PolynomialFeatures(degree=order)
			X_poly = poly_features.fit_transform(x)
			model = LinearRegression()
			model.fit(X_poly, y)
			y_pred = model.predict(X_poly)
			error = np.mean((y_pred - y) ** 2)

			if error < best_error:
				best_error = error
				best_order = order
				best_model = model

		def func(x_val):
			x_val_poly = poly_features.transform(np.array(x_val).reshape(-1, 1))
			return best_model.predict(x_val_poly)

		return lambda x_val: func(x_val)

test = polytrend()

data = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]

fitted_func = test.poly(data)

print(fitted_func(100.0))