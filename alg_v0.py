import numpy as np
import matplotlib.pyplot as plt

class polytrend:
	def __init__(self) -> None:
		pass

	# Performs polynomial trend analysis on the given data points.
	# Returns a function that predicts y values for given x values based on the best-fit polynomial equation.
	# Time complexity: O(n^3)
	# Space complexity: O(n)
	def poly(self, data):
		# Extract x and y coordinates from the data
		x = np.array([point[0] for point in data])
		y = np.array([point[1] for point in data])

		# Variables to store the best polynomial order, error, and coefficients
		best_order = -1
		best_error = float('inf')
		best_coeffs = np.array([])

		# Iterate over polynomial orders from 0 to 4
		for order in range(0, 5):
			# Perform polynomial fitting
			temp_coeffs = np.polyfit(x, y, order)

			# Calculate predicted y values using the fitted polynomial
			y_pred = np.polyval(temp_coeffs, x)

			# Calculate MSE between predicted y values and actual y values
			error = np.mean((y_pred - y) ** 2)

			# Update best order, error, and coefficients if the current error is lower
			if error < best_error:
				best_error = error
				best_order = order
				best_coeffs = temp_coeffs
			
			if order == 4:
				best_coeffs = np.polyfit(x, y, best_order)
				final_func = np.poly1d(best_coeffs)

		# return a lambda function
		return np.poly1d(best_coeffs)

	# Plots a graph of the polynomial function and distinguishes between known and extrapolated data.
	def poly_plot(self, data):
		# Extract x and y coordinates from the data
		x = np.array([point[0] for point in data])
		y = np.array([point[1] for point in data])

		# Find the maximum x value to determine the viewport width
		max_x = np.max(x)
		viewport_width = 1.5 * max_x

		# Obtain the best-fit polynomial function
		final_func = self.poly(data)

		# Generate points for plotting the polynomial function
		x_plot = np.linspace(min(x), viewport_width, 100)
		y_plot = final_func(x_plot)
		
		
		# Evaluate the polynomial function for the plot x values
		
		# Set the figure, title and axis labels
		plt.figure()
		plt.title('Polynomial Fit via PolyTrend')
		plt.xlabel('n')
		plt.ylabel('f(n)')
		
		# Plot the polynomial function
		plt.plot(x_plot, y_plot, label='Fitted Function')

		# Plot known data points
		plt.scatter(x, y, c='r', label='Known Data')

		# Plot extrapolated data points
		x_extrapolated = np.linspace(max_x, viewport_width, 100)
		y_extrapolated = final_func(x_extrapolated)
		plt.scatter(x_extrapolated, y_extrapolated, color='red', label='Extrapolated Data')

		# Add legend
		plt.legend()

		# Show the plot
		plt.show()

test = polytrend()

test_data = [
	(1, 1),
	(2, 2),
	(3, 3),
	(4, 4)
]

test.poly_plot(test_data)