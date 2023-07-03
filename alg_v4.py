# import PyQt6
# import PySide6
# import PyQt5
# import PySide2
import matplotlib
matplotlib.use('QtAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class polytrend:
	def __init__(self) -> None:
		pass

	# finding the best fit polynomial (between orders 0 and 4) of some data
	def polyfind(self, known_data, order=-1):
		# Extract x and y values from known_data
		x = np.array([x for x, _ in known_data]).reshape(-1, 1)
		y = np.array([y for _, y in known_data])

		if order == -1:
			# Find the best order polynomial between 0 and 4 using cross-validation
			best_order = -1
			best_score = float('-inf')

			for i in range(1, 5):
				# Generate polynomial features
				poly_features = PolynomialFeatures(degree=i, include_bias=False)
				x_poly = poly_features.fit_transform(x)

				# Fit linear regression model
				reg = LinearRegression()
				reg.fit(x_poly, y)

				# Calculate the score (R-squared) to evaluate the fit
				score = reg.score(x_poly, y)

				# Update best order if the current order has a higher score
				if score > best_score:
					best_score = score
					best_order = i

			order = best_order

		# Fit the best order polynomial
		poly_features = PolynomialFeatures(degree=order, include_bias=False)
		x_poly = poly_features.fit_transform(x)
		reg = LinearRegression()
		reg.fit(x_poly, y)

		# Return a lambda function for the predicted values based on the polynomial
		return lambda x_vals: reg.predict(poly_features.transform(np.array(x_vals).reshape(-1, 1)))

	# plots a function and data, with no further processing. visualization purposes only
	def graph(self, func, known_data=[], extrap_data=[]):
		# Extract x and y values from known_data
		x_known = np.array([x for x, _ in known_data])
		y_known = np.array([y for _, y in known_data])

		# Prepare extrapolation data
		x_extrap = np.array(extrap_data).reshape(-1, 1)
		y_extrap = func(x_extrap)
		
		# setting graph prerequisites
		plt.figure()
		plt.title('Fitted Function via PolyTrend')
		plt.xlabel('x')
		plt.ylabel('f(x)')

		# Plot known data and extrapolated data
		plt.scatter(x_known, y_known, color='blue', label='Known data')
		plt.plot(x_known+x_extrap, y_known+y_extrap, color='r', label='Fitted Function')
		plt.show()

	# combining the two
	def polyplot(self, data, extrap_data=[], order=-1):
		# Find the best-fit polynomial function
		func = self.polyfind(data, order)

		# Plot the function and data
		self.graph(func, data, extrap_data)

# Example usage
test = polytrend()

data = [
	(1, 1),
	(2, 4),
	(3, 9),
	(4, 16),
	(5, 25)
]

extrapolated = [6, 7, 8, 9, 10]

test.polyplot(data, extrapolated)