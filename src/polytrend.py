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
		- polyfind(): Finds the best-fit polynomial function.
		- polygraph(): Plots the function, known data, and extrapolated data.
	'''

	def _validate_data(self, degrees: List[int], known_data: Union[List[Tuple[float, float]], str]) -> None:
		'''
		Validate degrees and known_data to ensure they are not empty.

		Args:
			degrees (List[int]): List of polynomial degrees to consider.
			known_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.

		Raises:
			ValueError: If degrees or known_data is empty.
		'''
		if not known_data:
			raise ValueError('Known data must be specified as a non-empty list or CSV file path.')

		if not degrees:
			raise ValueError('Degrees must be specified as a non-empty list.')

	def _read_known_data(self, main_data: Union[List[Tuple[float, float]], str]) -> List:
		"""
		Read and extract known data from either a list of tuples or a CSV file.

		Args:
			known_data (Union[List[Tuple[float, float]], str]): Known data points or CSV file path.

		Returns:
			List[str, str, str, List[(np.array, np.array)]]
		"""
		title = x_label = y_label = ''
		x_main = y_main = []

		if isinstance(main_data, str):
			# Read data from CSV file
			df = pd.read_csv(main_data)
			x_label = df.columns[0]
			y_label = df.columns[1]
			x_main = df[x_label].values
			y_main = df[y_label].values
			title = f'Fitted Function via PolyTrend - Data from {main_data}'

		elif isinstance(main_data, list):
			# Unpack list of tuples into separate x_main and y_data arrays
			x_label = 'x'
			y_label = 'f(x)'
			x_main, y_main = zip(*main_data)
			title = f'Fitted Function via PolyTrend'

		else:
			raise ValueError('Data must be a non-empty list of tuples or CSV file path')

		return [title, x_label, y_label, [(np.array(x_main), np.array(y_main))]]

	def polyplot(self,
		degrees: List[int],
		known_data: Union[List[Tuple[float, float]], str],
		extrap_data: List[float] = [],
		savefigure: bool = False
	) -> None:
		'''
		Plot the polynomial fit on the known data.

		Args:
			degrees (List[int]): List of polynomial degrees to consider.
			known_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.
			extrap_data (List[float], optional): List of x coordinates for extrapolation. Defaults to [].
			savefigure (bool, optional): Whether to save the figure as a PNG. Defaults to False.

		Raises:
			ValueError: If degrees and/or known data is not specified or empty.
		'''
		self._validate_data(degrees, known_data)
		fitted_model = self.polyfind(degrees, known_data)
		self.polygraph(known_data, extrap_data, func=fitted_model, savefigure=savefigure)

	def polyfind(self,
		degrees: List[int],
		known_data: Union[List[Tuple[float, float]], str]
	) -> Callable[[List[float]], np.ndarray]:
		'''
		Find the best-fit polynomial function.

		Args:
			degrees (List[int]): List of polynomial degrees to consider.
			known_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.

		Returns:
			Callable[[List[float]], np.ndarray]: A function that predicts values based on the polynomial.
		'''
		self._validate_data(degrees, known_data)
		temp_list = self._read_known_data(main_data=known_data)
		x_main, y_main = zip(*temp_list[3])

		# Initialize variables to store the best polynomial features, regressor, and score
		best_poly_features = None
		best_reg = None
		best_score = float('-inf')

		# Loop through each degree to find the best-fit polynomial
		for degree in degrees:
			# Create polynomial features up to the current degree
			poly_features = PolynomialFeatures(degree=degree, include_bias=False)
			x_poly = poly_features.fit_transform(x_main.reshape(-1, 1))

			# Initialize a Linear Regression model and fit it to the polynomial features
			reg = LinearRegression()
			reg.fit(x_poly, y_main.reshape(-1, 1))

			# Calculate the R-squared score of the regression
			score = reg.score(x_poly, y_main.reshape(-1, 1))

			# Update the best values if the current score is higher
			if score > best_score:
				best_score = score
				best_poly_features = poly_features
				best_reg = reg

		if best_reg is None or best_poly_features is None:
			raise ValueError('All degrees had a lower score than negative infinity. Something\'s up with that data source...')
		else:
			print(f'Best R-squared score for given degree range: {best_score}')
			return lambda x_vals: best_reg.predict(best_poly_features.transform(np.array(x_vals).reshape(-1, 1))).flatten()

	def polygraph(self,
		known_data: Union[List[Tuple[float, float]], str],
		extrap_data: List[float] = [],
		func: Optional[Callable[[List[float]], np.ndarray]] = None,
		savefigure: bool = False
	) -> None:
		'''
		Plot the function, known data, and extrapolated data.

		Args:
			known_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.
			extrap_data (List[float], optional): List of extrapolation data points. Defaults to [].
			func (Optional[Callable[[List[float]], np.ndarray]], optional): Function to generate predicted values. Defaults to None.
			savefigure (bool, optional): Whether to save the figure as a PNG. Defaults to False.

		Raises:
			ValueError: If known data is empty.
			ValueError: If extrapolation data is provided but no function is given.
		'''
		if not known_data:
			raise ValueError('Known data must be specified as a non-empty list or CSV file path.')

		if extrap_data and not func:
			raise ValueError("If extrapolation data is provided, a function to generate predicted values must also be given.")

		# Making the graph figure
		plt.figure()

		# Reading (and extracting from) the known data
		temp_list = self._read_known_data(known_data)
		title = temp_list[0]
		x_label, y_label = temp_list[1], temp_list[2]
		x_main, y_data = zip(*temp_list[3])

		# Setting prereqs
		plt.title(title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.plot(x_main, y_data, color='blue', label='Known Data')

		if func:
			# Plot the fitted function if a function is provided
			x_func = np.linspace(min(x_main), max(x_main), 100)
			y_func = func(list(x_func))
			plt.plot(x_func, y_func, color='green', label='Fitted Function')

		if extrap_data and func:
			# If both extrapolation data and function are provided
			# Generate and plot extrapolated data
			x_extrap = np.array(extrap_data)
			y_extrap = func(list(x_extrap))
			plt.scatter(x_extrap, y_extrap, color='red', label='Extrapolated data')

		# Add legend and display the plot
		plt.legend()

		if not savefigure:
			# Display the plot if savefigure is not requested
			plt.show()
		else:
			# Save the plot as PNG with timestamp in the filename
			timestamp = datetime.now().strftime('%Y.%m-%d-%H:%M')
			filename = f'plot_{timestamp}_{random.randint(0,1000)}.png'
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
	known_data = [(1, 2), (2, 4), (3, 9)]
	poly_trend = PolyTrend()

	try:
		poly_trend.polyplot(degrees, known_data, extrap_data=[4, 5], savefigure=True)
	except ValueError as ve:
		print(f"Error: {ve}")

	print("Example usage completed.")
