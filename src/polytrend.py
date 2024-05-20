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

from random import uniform
from datetime import datetime
from typing import Union, List, Tuple, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from csv import reader
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


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

    def _read_main_data(self, main_data: Union[List[Tuple[float, float, float]], str]) -> List:
        """
        Read and extract known data from either a list of tuples or a CSV file.

        Args:
            main_data (Union[List[Tuple[float, float, float]], str]): Known data points or CSV file path.

        Returns:
            List: [str, str, str, List[Tuple[float, float, float]]]
        """
        graph_title = x_axis_label = y_axis_label = str()
        x_main_values = y_main_values = err_values = []

        if isinstance(main_data, str):
            with open(main_data, mode='r') as file:
                reader_var = reader(file)
                headers = next(reader_var)
                # this doesn't work right now
                x_axis_label = headers[0].strip()
                y_axis_label = headers[1].strip()
                x_main_values = []
                y_main_values = []
                for row in reader_var:
                    x_main_values.append(row[0])
                    y_main_values.append(row[1])
            graph_title = f"Fitted Function via PolyTrend - Data from {main_data}"

        elif isinstance(main_data, list):
            # Unpack list of tuples into separate x_main and y_data arrays
            x_axis_label = "x"
            y_axis_label = "f(x)"
            x_main_values, y_main_values, err_values = zip(*main_data)
            graph_title = f"Fitted Function via PolyTrend"

        else:
            raise ValueError("Data must be a non-empty list of tuples or CSV file path")

        data_points = [
            (float(x), float(y), float(e)) for x, y, e in zip(x_main_values, y_main_values, err_values)
        ]

        return [graph_title, x_axis_label, y_axis_label, data_points]

    def polyplot(
        self,
        degrees: List[int],
        main_data: Union[List[Tuple[float, float, float]], str],
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
        self.polygraph(main_data, extrapolate_data, function=fitted_model[0])

    def polyfind(
        self, degrees: List[int], main_data: Union[List[Tuple[float, float, float]], str]
    ) -> Tuple[Callable[[list], np.ndarray], int]:
        """
        Find the best-fit polynomial function.

        Args:
            degrees (List[int]): List of polynomial degrees to consider.
            main_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.

        Returns:
                Callable[[List[float]], List[float]]: A function that predicts values based on the polynomial.
        """

        # ! broken, needs update
        def _construct_polynomial_expression(coefficients: np.ndarray, intercept: np.ndarray):
            processed_coeffs = coefficients[0].tolist()[::-1]
            processed_intercept = intercept.tolist()
            polynomial = ""

            for i in range(0, len(processed_coeffs)):
                # rounding is messy for really small coeffs
                coeff_rounded = round(processed_coeffs[i], 3)
                term = f"({abs(coeff_rounded)})x^{len(processed_coeffs) - i}"
                if i == 0:
                    polynomial += f"({coeff_rounded})x^{len(processed_coeffs)}" + " "
                else:
                    sign = "+ " if coeff_rounded > 0 else "- "
                    polynomial += sign + term + " "

            # might be an error in indexing; could be 1D array
            if processed_intercept[0] < 0:
                polynomial += f"- {abs(round(processed_intercept[0], 3))}"
            else:
                polynomial += f"+ {round(processed_intercept[0], 3)}"

            return polynomial


        # * switched to ridge regression (see README)
        def _model_selector(
            x_main: np.ndarray, y_main: np.ndarray, errors: np.ndarray
        ) -> Tuple[Union[Ridge, LinearRegression], PolynomialFeatures, float]:
            # TODO there might be an optimization opportunity. look into other regression libs & algorithms

            best_bic = float("inf")
            best_model = None
            best_poly_features = None
            best_r2_score = None

            for degree in degrees:
                poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                x_poly = poly_features.fit_transform(x_main)

                # there might be an inherent accuracy discrepancy between Ridge and OSE. see base.csv
                if np.any(errors != 0):
                    # TODO implement cross validation to avoid arbitrary alpha values
                    reg = Ridge(alpha=1.0)
                    reg.fit(x_poly, y_main, sample_weight=1 / errors.ravel())
                else:
                    reg = LinearRegression()
                    reg.fit(x_poly, y_main)

                # Predicting y values with the current model
                y_pred = reg.predict(x_poly)

                # Coefficient of determination
                r2 = r2_score(y_main, y_pred)

                # Bayesian Information Criterion metric
                n = len(y_main)
                k = degree + 1
                mse = mean_squared_error(y_main, y_pred)
                bic_score = n * np.log(mse) + k * np.log(n)

                if bic_score < best_bic:
                    best_bic = bic_score
                    best_model = reg
                    best_poly_features = poly_features
                    best_r2_score = r2

            if best_model is None or best_poly_features is None:
                raise ValueError("All models had a BIC score of positive infinity... 何？！")

            return best_model, best_poly_features, best_bic, best_r2_score

        processed_data = self._read_main_data(main_data=main_data)
        x_main, y_main, err_main = zip(*processed_data[3])
        # ! possible redundancy in reshaping
        x_main = np.asarray(x_main).reshape(-1, 1)
        y_main = np.asarray(y_main).reshape(-1, 1)
        err_main = np.asarray(err_main).reshape(-1, 1)

        best_model, best_poly_features, best_bic, r_value = _model_selector(x_main, y_main, err_main)

        coefficients = best_model.coef_
        intercept = best_model.intercept_

        global func_expression
        func_expression = _construct_polynomial_expression(coefficients, intercept)

        print(
            f"{datetime.now().strftime('%Y.%m-%d-%H:%M')}\nGenerated function: {func_expression}\nBest BIC score for given degree range: {best_bic}\nCoefficient of determination (r-squared value): {r_value}"
        )

        return (
            lambda x_vals: best_model.predict(
                best_poly_features.transform(np.array(x_vals).reshape(-1, 1))
            ).flatten(),
            len(coefficients) - 1,
        )  # Assuming coefficients include the intercept

    def polygraph(
        self,
        main_data: Union[List[Tuple[float, float, float]], str],
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
        title, x_label, y_label = processed_data[0], processed_data[1], processed_data[2]
        x_main, y_main, err_main = zip(*processed_data[3])

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if all(x == 0 for x in err_main):
            plt.scatter(x_main, y_main, color="blue", label="Known Data")
        plt.errorbar(x_main, y_main, yerr=err_main, label="Known Data", fmt='o', capsize=5)

        if function:
            x_func = np.linspace(min(x_main), max(x_main), 100)
            y_func = function(list(x_func))
            plt.plot(x_func, y_func, color="green", label=f"{func_expression}")
            
            # * RESIDUAL ANALYSIS: plots a 2nd graph, not relevant to HS
            # func_predictions = function(list(x_main))
            # y_calculated_err = np.abs(np.array([y_main]) - np.array([func_predictions]))

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
                    plt.annotate(
                        f"({x:.2f}, {y:.2f})",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor="black",
                            linewidth=0.5,
                        ),
                    )  # Adding a white background

        plt.grid(True)    
        plt.legend()
        plt.show()


if __name__ == "__main__":
    degrees = [1, 2, 3]
    data = [
        (float(x), float(0.5 * x**2 - 2 * x + 1 + uniform(-1000, 1000)), 1.0)
        for x in range(0, 100)
    ]
    polytrend_instance = PolyTrend()

    try:
        polytrend_instance.polyplot(degrees, data, extrapolate_data=[15, 20])
    except ValueError as ve:
        print(f"Error: {ve}")

    print("Example usage completed.")
