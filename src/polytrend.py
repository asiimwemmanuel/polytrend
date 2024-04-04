# This file is part of PolyTrend.
#
# PolyTrend is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PolyTrend is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PolyTrend. If not, see <https://www.gnu.org/licenses/>.

import random
from datetime import datetime
from typing import Union, List, Tuple, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


class PolyTrend:
	"""
	PolyTrend: A utility for polynomial trend fitting, visualization, and extrapolation.

	This class provides methods for finding the best-fit polynomial function for a given set of known data points,
	plotting the function along with the known data points and extrapolated data points.

	Main Methods:
		- polyplot(): Finds and plots the best polynomial fit on the known data.
		- polyfind(): Finds the best-fit polynomial function.z
		- polygraph(): Plots the function, known data, and extrapolated data.
	"""

	def _read_main_data(self, main_data: Union[List[Tuple[float, float]], str]) -> List:
		"""
		Read and extract known data from either a list of tuples or a CSV file.

		Args:
		main_data (Union[List[Tuple[float, float]], str]): Known data points or CSV file path.

		Returns:
			List: [str, str, str, List[Tuple[float, float]]]
		"""
		graph_title = x_axis_label = y_axis_label = ""
		x_main_values = y_main_values = []

		if isinstance(main_data, str):
			data_frame = pd.read_csv(main_data)
			x_axis_label = data_frame.columns[0]
			y_axis_label = data_frame.columns[1]
			x_main_values = data_frame[x_axis_label].values
			y_main_values = data_frame[y_axis_label].values
			graph_title = f"Fitted Function via PolyTrend - Data from {main_data}"

		elif isinstance(main_data, list):
			# Unpack list of tuples into separate x_main and y_data arrays
			x_axis_label = "x"
			y_axis_label = "f(x)"
			x_main_values, y_main_values = zip(*main_data)
			graph_title = f"Fitted Function via PolyTrend"

		else:
			raise ValueError("Data must be a non-empty list of tuples or CSV file path")

		data_points = [
			(float(x), float(y)) for x, y in zip(x_main_values, y_main_values)
		]

		return [graph_title, x_axis_label, y_axis_label, data_points]

	def polyplot(
		self,
		degrees: List[int],
		main_data: Union[List[Tuple[float, float]], str],
		extrapolate_data: List[float] = [],
	) -> None:
		"""
		Plot the polynomial fit on the known data.

		Args:
			degrees (List[int]): List of polynomial degrees to consider.
			main_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.
			extrapolate_data (List[float], optional): List of x coordinates for extrapolation. Defaults to [].

		Raises:
			ValueError: If degrees and/or known data is not specified or empty.
		"""
		fitted_model = self.polyfind(degrees, main_data)
		self.polygraph(
			main_data, extrapolate_data, function=fitted_model[0]
		)

	def polyfind(
		self, degrees: List[int], main_data: Union[List[Tuple[float, float]], str]
	) -> Tuple[Callable[[list], np.ndarray], int]:
		"""
		Find the best-fit polynomial function.

		Args:
			degrees (List[int]): List of polynomial degrees to consider.
			main_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.

		Returns:
			Callable[[List[float]], List[float]]: A function that predicts values based on the polynomial.
		"""

		def _construct_polynomial_expression(coefficients, intercept):
			expression = f"f(x) = {intercept:.3g} " + " ".join([f"+ ({coef:.3g})x^{i + 1}" for i, coef in enumerate(coefficients)])
			return expression
		
		def _model_selector(
			x_main: np.ndarray, y_main: np.ndarray
		) -> Tuple[LinearRegression, PolynomialFeatures, float]:

			buffer_bic = float("inf")
			buffer_model = None
			buffer_poly_features = None

			for degree in degrees:
				# the X matrix
				poly_features = PolynomialFeatures(degree=degree, include_bias=False)
				x_poly = poly_features.fit_transform(x_main)

				# the Normal Equation in action
				reg = LinearRegression()
				reg.fit(x_poly, y_main)

				# Bayesian Information Criterion metric
				n = len(y_main)
				k = degree + 1
				mse = mean_squared_error(y_main, reg.predict(x_poly))
				bic_score = n * np.log(mse) + k * np.log(n)

				if bic_score < buffer_bic:
					buffer_bic = bic_score
					buffer_model = reg
					buffer_poly_features = poly_features

			if buffer_model is None or buffer_poly_features is None:
				raise ValueError("All models had a BIC score of positive infinity... 何？！")

			return buffer_model, buffer_poly_features, buffer_bic

		processed_data = self._read_main_data(main_data=main_data)
		x_main, y_main = zip(*processed_data[3])
		x_main = np.asarray(x_main).reshape(-1, 1)
		y_main = np.asarray(y_main)

		best_fit_model, best_poly_features, best_bic = _model_selector(x_main, y_main)

		coefficients = best_fit_model.coef_
		intercept = best_fit_model.intercept_

		global func_expression
		func_expression = _construct_polynomial_expression(coefficients, intercept)

		print(f"{datetime.now().strftime('%Y.%m-%d-%H:%M')}\nGenerated function: {func_expression}\nBest BIC score for given degree range: {best_bic}")

		# return lambda x_vals: best_fit_model.predict(
		# 	best_poly_features.transform(np.array(x_vals).reshape(-1, 1))
		# ).flatten()

		return lambda x_vals: best_fit_model.predict(
			best_poly_features.transform(np.array(x_vals).reshape(-1, 1))
		).flatten(), len(coefficients) - 1  # Assuming coefficients include the intercept

	def polygraph(
		self,
		main_data: Union[List[Tuple[float, float]], str],
		extrapolate_data: List[float] = [],
		function: Optional[Callable[[List[float]], np.ndarray]] = None,
	) -> None:
		"""
		Plot the function, known data, and extrapolated data.

		Args:
			main_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.
			extrapolate_data (List[float], optional): List of extrapolation data points. Defaults to [].
			func (Optional[Callable[[List[float]], np.ndarray]], optional): Function to generate predicted values. Defaults to None.

		Raises:
			ValueError: If known data is empty.
			ValueError: If extrapolation data is provided but no function is given.
		"""
		if extrapolate_data and function is None:
			raise ValueError(
				"If extrapolation data is provided, a function to generate predicted values must also be given."
			)

		plt.figure()

		processed_data = self._read_main_data(main_data)
		title = processed_data[0]
		x_label, y_label = processed_data[1], processed_data[2]
		x_main, y_main = zip(*processed_data[3])

		plt.title(title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.scatter(x_main, y_main, color="blue", label="Known Data")

		if function:
			x_func = np.linspace(min(x_main), max(x_main), 100)
			y_func = function(list(x_func))
			plt.plot(x_func, y_func, color="green", label=f"{func_expression}")

			if extrapolate_data:
				# Extract and plot extrapolated data
				y_extrap = function(list(extrapolate_data))

				# extending the function line
				x_func_extension = np.linspace(max(x_main), max(extrapolate_data), 100)
				y_func_extension = function(list(x_func_extension))
				plt.plot(x_func_extension, y_func_extension, color="green")

				plt.scatter(
					extrapolate_data, y_extrap, color="red", label="Extrapolated data"
				)

				# Label each extrapolated point with its coordinates
				for x, y in zip(extrapolate_data, y_extrap):
					plt.annotate(f"({x:.2f}, {y:.2f})", 
								(x, y), 
								textcoords="offset points", 
								xytext=(0,10), 
								ha='center',
								bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=0.5)) # Adding a white background

		plt.legend()
		plt.show()

if __name__ == "__main__":
	degrees = [1, 2, 3]
	data = [
		(float(x), float(0.5 * x**2 - 2 * x + 1 + random.uniform(-1000, 1000)))
		for x in range(0, 100)
	]
	polytrend_instance = PolyTrend()

	try:
		polytrend_instance.polyplot(
			degrees, data, extrapolate_data=[15, 20]
		)
	except ValueError as ve:
		print(f"Error: {ve}")

	print("Example usage completed.")
