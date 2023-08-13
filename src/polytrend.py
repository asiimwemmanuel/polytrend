import os
import random
from datetime import datetime
from typing import Union, List, Tuple, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class PolyTrend:
	'''
	PolyTrend: A utility for polynomial trend fitting, visualization, and extrapolation.

	This class provides methods for finding the best-fit polynomial function for a given set of known data points,
	plotting the function along with the known data points and extrapolated data points, and saving the plots as PNG images if specified.

	Main Methods:
		- polyplot(): Finds and plots the best polynomial fit on the known data.
		- polyfind(): Finds the best-fit polynomial function.z
		- polygraph(): Plots the function, known data, and extrapolated data.
	'''

	def _validate_data(self,
			degrees: List[int],
			main_data: Union[List[Tuple[float, float]], str]
		) -> None:
		'''
		Validate degrees and main_data to ensure they are not empty.

		Args:
			degrees (List[int]): List of polynomial degrees to consider.
			main_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.

		Raises:
			ValueError: If degrees or main_data is empty.
		'''
		if not main_data:
			raise ValueError('Main data must be specified as a non-empty list or CSV file path.')

		if not degrees:
			raise ValueError('Degrees must be specified as a non-empty list.')

	def _read_main_data(self,
			main_data: Union[List[Tuple[float, float]], str]
		) -> List:
		"""
		Read and extract known data from either a list of tuples or a CSV file.

		Args:
			main_data (Union[List[Tuple[float, float]], str]): Known data points or CSV file path.

		Returns:
			List: [str, str, str, List[Tuple[float, float]]]
		"""
		graph_title = x_axis_label = y_axis_label = ''
		x_main_values = y_main_values = []

		if isinstance(main_data, str):
			# Read data from CSV file
			data_frame = pd.read_csv(main_data)
			x_axis_label = data_frame.columns[0]
			y_axis_label = data_frame.columns[1]
			x_main_values = data_frame[x_axis_label].values
			y_main_values = data_frame[y_axis_label].values
			graph_title = f'Fitted Function via PolyTrend - Data from {main_data}'

		elif isinstance(main_data, list):
			# Unpack list of tuples into separate x_main and y_data arrays
			x_axis_label = 'x'
			y_axis_label = 'f(x)'
			x_main_values, y_main_values = zip(*main_data)
			graph_title = f'Fitted Function via PolyTrend'

		else:
			raise ValueError('Data must be a non-empty list of tuples or CSV file path')

		data_points = [(float(x), float(y)) for x, y in zip(x_main_values, y_main_values)]

		return [graph_title, x_axis_label, y_axis_label, data_points]


	def polyplot(self,
		degrees: List[int],
		main_data: Union[List[Tuple[float, float]], str],
		extrapolate_data: List[float] = [],
		save_figure: bool = False
	) -> None:
		'''
		Plot the polynomial fit on the known data.

		Args:
			degrees (List[int]): List of polynomial degrees to consider.
			main_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.
			extrapolate_data (List[float], optional): List of x coordinates for extrapolation. Defaults to [].
			savefigure (bool, optional): Whether to save the figure as a PNG. Defaults to False.

		Raises:
			ValueError: If degrees and/or known data is not specified or empty.
		'''
		self._validate_data(degrees, main_data)
		fitted_model = self.polyfind(degrees, main_data)
		self.polygraph(main_data, extrapolate_data, function=fitted_model, save_figure=save_figure)

	def polyfind(self,
		degrees: List[int],
		main_data: Union[List[Tuple[float, float]], str]
	) -> Callable[[List[float]], np.ndarray]:
		'''
		Find the best-fit polynomial function.

		Args:
			degrees (List[int]): List of polynomial degrees to consider.
			main_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.

		Returns:
			Callable[[List[float]], List[float]]: A function that predicts values based on the polynomial.
		'''
		self._validate_data(degrees=degrees, main_data=main_data)

		# shaping the data into usable structures
		processed_data = self._read_main_data(main_data=main_data)
		x_main, y_main = zip(*processed_data[3])
		x_main = np.asarray(x_main).reshape(-1, 1)
		y_main = np.asarray(y_main).reshape(-1, 1)

		# preallocate memory for best polynomial features and regression model
		POLY_FEATURES = None
		REGRESSOR = None

		# performance tracker (R-squared measure)
		BEST_R_SQUARED = float('-inf')

		# find the best-fit polynomial function; Brute Force approach
		for degree in degrees:
			# generate polynomial features
			poly_features = PolynomialFeatures(degree=degree, include_bias=False)
			x_poly = poly_features.fit_transform(x_main)

			# fit linear regression model
			reg = LinearRegression()
			reg.fit(x_poly, y_main)

			# calculate score for model evaluation
			score = reg.score(x_poly, y_main)

			# update the best score and models if a higher score is obtained
			if score > BEST_R_SQUARED:
				BEST_R_SQUARED = score
				POLY_FEATURES = poly_features
				REGRESSOR = reg

		if REGRESSOR is None or POLY_FEATURES is None:
			raise ValueError('All models had a score below negative infinity...Dayum, boi!')
		else:
			print(f'Best R-squared score for given degree range: {BEST_R_SQUARED}')

			# Printing the best-fit polynomial function
			coefficients = REGRESSOR.coef_
			intercept = np.array(REGRESSOR.intercept_)
			# Nested helper function to construct the polynomial expression
			def construct_polynomial_expression(coefficients):
				expression = f'f(x) = '
				for i, coef in enumerate(coefficients):
					if i > 0:
						expression += ' + '  # Add "+" before coefficients other than the first one
					if abs(coef) >= 1e-8:
						expression += f'({coef:.4f})x^{i}'
				return expression

			# Print the constructed polynomial expression
			coefficients = REGRESSOR.coef_
			print(construct_polynomial_expression(coefficients))

			return lambda x_vals: REGRESSOR.predict(POLY_FEATURES.transform(np.array(x_vals).reshape(-1, 1))).flatten()

	def polygraph(self,
		main_data: Union[List[Tuple[float, float]], str],
		extrapolate_data: List[float] = [],
		function: Optional[Callable[[List[float]], np.ndarray]] = None,
		save_figure: bool = False
	) -> None:
		'''
		Plot the function, known data, and extrapolated data.

		Args:
			main_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.
			extrapolate_data (List[float], optional): List of extrapolation data points. Defaults to [].
			func (Optional[Callable[[List[float]], np.ndarray]], optional): Function to generate predicted values. Defaults to None.
			savefigure (bool, optional): Whether to save the figure as a PNG. Defaults to False.

		Raises:
			ValueError: If known data is empty.
			ValueError: If extrapolation data is provided but no function is given.
		'''
		if not main_data:
			raise ValueError('Known data must be specified as a non-empty list or CSV file path.')

		if extrapolate_data and not function:
			raise ValueError("If extrapolation data is provided, a function to generate predicted values must also be given.")

		# Making the graph figure
		plt.figure()

		# Reading (and extracting from) the known data
		temp_list = self._read_main_data(main_data)
		title = temp_list[0]
		x_label, y_label = temp_list[1], temp_list[2]
		x_main, y_main = zip(*temp_list[3])

		# Setting prereqs
		plt.title(title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.scatter(x_main, y_main, color='blue', label='Known Data')

		if function:
			x_func = np.linspace(min(x_main), max(x_main), 100)
			y_func = function(list(x_func))
			plt.plot(x_func, y_func, color='green', label='Fitted Function')

			if extrapolate_data:
				# Extract and plot extrapolated data
				# extrapolate_data = np.array(extrapolate_data)
				y_extrap = function(list(extrapolate_data))

				# extending the function line
				x_func_extension = np.linspace(max(x_main), max(extrapolate_data), 100)
				y_func_extension = function(list(x_func_extension))
				plt.plot(x_func_extension, y_func_extension, color='green')

				plt.scatter(extrapolate_data, y_extrap, color='red', label='Extrapolated data')

		# Add legend and display the plot
		plt.legend()

		if not save_figure:
			# Display the plot if savefigure is not requested
			plt.show()
		else:
			# Save the plot as PNG with timestamp in the filename
			timestamp = datetime.now().strftime('%Y.%m-%d-%H:%M')
			filename = f'plot_{timestamp}_{random.randint(0,10)}.png'
			figures_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
			os.makedirs(figures_dir, exist_ok=True)
			filepath = os.path.join(figures_dir, filename)

			try:
				# Save the plot to the specified filepath
				plt.savefig(filepath)
				print(f'Figure saved successfully: {filepath}')
			except Exception as e:
				print(f'Error saving figure: {str(e)}')

if __name__ == '__main__':
	degrees = [1, 2, 3, 4]
	data = [(float(x), float(x**2)) for x in range(-10, 11)]
	polytrend_instance = PolyTrend()

	try:
		polytrend_instance.polyplot(degrees, data, extrapolate_data=[4, 5], save_figure=False)
	except ValueError as ve:
		print(f"Error: {ve}")

	print("Example usage completed.")
