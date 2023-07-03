import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LineaRegression
# from sklearn.preprocessing import PolynomialFeatures

class polytrend:
	def __init__(self) -> None:
		pass

	# finding the best fit polynomial (between orders 0 and 4) of some data
	def polyfind(self, known_data) -> np.poly1d:
		# extracting data points into useful data structures
		# lists MUST be the same size
		x = np.array([point[0] for point in known_data]).reshape(-1, 1)
		y = np.array([point[1] for point in known_data])

		# declaring measures of performance
		best_mse = float('inf') # could also use Bayesian Information Criterion
		unfiltered_coeffs = np.array([])

		# Creating polynomial features
		# determining performance of models 0 through 4 inclusive via MSE
		for order in range(5):
		# 	# coeffs = np.polyfit(x, y, order)
		# 	# y_pred = np.polyval(coeffs, x)
		# 	error = mean_squared_error(y, y_pred)

		# 	if error < best_error:
		# 		best_error = error
		# 		unfiltered_coeffs = coeffs
			polynomial_features = PolynomialFeatures(degree=order)
			x_poly = polynomial_features.fit_transform(x)

			# Fitting the polynomial regression model
			model = LinearRegression()
			model.fit(x_poly, y)

			# Predicting y-values using the model
			y_pred = model.predict(x_poly)

			# Calculating mean squared error
			mse = mean_squared_error(y, y_pred)

			# Updating the best degree if current degree performs better
			if mse < best_mse:
				best_degree = order
				best_mse = mse

		# coefficient comparator
		def coeff_comp(coeffs):
			total_sum = np.sum(np.abs(coeffs))
			contributions = np.abs(coeffs) / total_sum
			return contributions

		# threshold coefficient remover
		def coeff_remover(coeffs, threshold=0.01):
			contributions = coeff_comp(coeffs)
			filtered_coeffs = np.where(contributions < threshold, 0.0, coeffs)
			return filtered_coeffs

		# filtering the important coeffs
		filtered_coeffs = coeff_remover(unfiltered_coeffs)

		# rounding the coefficients
		filtered_coeffs = np.round(filtered_coeffs, decimals=2)

		# removing trailing & leading 0's
		def trim_zeros(arr):
			nonzero_indices = np.nonzero(arr)
			start = nonzero_indices[0][0]
			end = nonzero_indices[0][-1] + 1
			return arr[start:end]

			return arr[start:end]
		
		filtered_coeffs = trim_zeros(filtered_coeffs)

		polynomial = np.poly1d(filtered_coeffs)

		print(polynomial)

		return polynomial

	# plots a function and data, with no further processing. visualisation purposes only
	def graph(self, func, known_data=[], extrap_data=[]) -> None:
		# Extracting known data into useful structures
		x = np.array([point[0] for point in known_data])
		y = np.array([point[1] for point in known_data])
	
		# Combining x, y, and extrap_data to determine the scale
		all_data = np.concatenate([x, y, extrap_data])
		data_min = np.min(all_data)
		data_max = np.max(all_data)
	
		# Setting the window size with the same scale for both axes
		viewport_width = 1.2 * (data_max - data_min)
	
		# Modelling the function plot
		x_plot = np.linspace(data_min, viewport_width, 100)
		y_plot = func(x_plot)
	
		# Setting graph prerequisites
		plt.figure()
		plt.title('Fitted Function via PolyTrend')
		plt.xlabel('x')
		plt.ylabel('f(x)')
	
		# Plotting the function as well as known data
		plt.plot(x_plot, y_plot, label='Fitted Function')
		plt.scatter(x, y, color='blue', label='Known Data')
	
		# Plotting extrapolates, if there are any
		if extrap_data != []:
			plt.scatter(extrap_data, func(extrap_data), color='red', label='Calculated Data')
	
		# Setting the same scale for both axes
		plt.axis('equal')
	
		# Finalizing the graph
		plt.legend()
		plt.show()


	# combining the two
	def polyplot(self, data, extrap=[], order=-1) -> None:
		if order != -1:
			x = np.array([point[0] for point in known_data])
			y = np.array([point[1] for point in known_data])
			coeffs = np.polyfit(x, y, order)
			polynomial = np.poly1d(coeffs, extrap)
			self.graph(polynomial, data, extrap)
		else:
			polynomial = self.polyfind(data)
			self.graph(polynomial, data, extrap)

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