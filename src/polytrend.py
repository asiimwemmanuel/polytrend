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
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

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
            Find the best-fit polynomial function with comprehensive statistical evaluation.

            Args:
                degrees (List[int]): List of polynomial degrees to consider.
                main_data (Union[List[Tuple[float, float]], str]): List of tuples representing the known data points or CSV file path.

            Returns:
                Tuple containing:
                    - Callable[[List[float]], List[float]]: A function that predicts values based on the polynomial
                    - int: The degree of the selected polynomial
                Also prints comprehensive statistical measures of the fit.
            """

            def _construct_polynomial_expression(coefficients: np.ndarray, intercept: np.ndarray) -> str:
                """Constructs a human-readable polynomial equation string."""
                coefficients_list = np.asarray(coefficients).tolist()[::-1]
                intercept_list = intercept.tolist()
                polynomial = ""
                
                for i, coeff in enumerate(coefficients_list):
                    rounded_coeff = round(float(coeff), 3)
                    term = f"({abs(rounded_coeff)})x^{len(coefficients_list) - i}"
                    if i == 0:
                        polynomial += f"({rounded_coeff})x^{len(coefficients_list)}" + " "
                    else:
                        sign = "+ " if rounded_coeff > 0 else "- "
                        polynomial += sign + term + " "

                if intercept_list[0] < 0:
                    polynomial += f"- {abs(round(intercept_list[0], 3))}"
                else:
                    polynomial += f"+ {round(intercept_list[0], 3)}"

                return polynomial

            def _calculate_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> dict:
                """Calculate various statistical metrics for model evaluation."""
                n = len(y_true)
                residuals = y_true - y_pred
                
                metrics = {
                    'mse': mean_squared_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred),
                    'adj_r2': 1 - (1 - r2_score(y_true, y_pred)) * (n - 1) / (n - n_params - 1),
                    'aic': n * np.log(mean_squared_error(y_true, y_pred)) + 2 * n_params,
                    'bic': n * np.log(mean_squared_error(y_true, y_pred)) + n_params * np.log(n),
                    'residual_stats': {
                        'mean': np.mean(residuals),
                        'std': np.std(residuals),
                        'min': np.min(residuals),
                        'max': np.max(residuals)
                    }
                }
                return metrics

            def _model_selector(
                x_main: np.ndarray, y_main: np.ndarray, errors: np.ndarray
            ) -> Tuple[Union[Ridge, LinearRegression], PolynomialFeatures, dict]:
                """Selects the best model based on multiple statistical criteria."""
                
                best_metrics = {
                    'bic': float('inf'),
                    'aic': float('inf'),
                    'adj_r2': -float('inf'),
                    'model': None,
                    'poly_features': None,
                    'degree': None
                }

                for degree in degrees:
                    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                    x_poly = poly_features.fit_transform(x_main)

                    if np.any(errors != 0):
                        reg = Ridge(alpha=1.0)
                        reg.fit(x_poly, y_main, sample_weight=1 / errors.ravel())
                    else:
                        reg = LinearRegression()
                        reg.fit(x_poly, y_main)

                    y_pred = reg.predict(x_poly)
                    n_params = x_poly.shape[1]  # Number of features (parameters)
                    
                    current_metrics = _calculate_additional_metrics(y_main, y_pred, n_params)
                    current_metrics.update({
                        'model': reg,
                        'poly_features': poly_features,
                        'degree': degree
                    })

                    # Update best model if current model has better BIC (primary criterion)
                    if current_metrics['bic'] < best_metrics['bic']:
                        best_metrics = current_metrics

                if best_metrics['model'] is None:
                    raise ValueError("All models had a BIC score of positive infinity... 何？！")

                return best_metrics

            # Process input data
            processed_data = self._read_main_data(main_data=main_data)
            x_main, y_main, err_main = zip(*processed_data[3])
            x_main = np.asarray(x_main).reshape(-1, 1)
            y_main = np.asarray(y_main).reshape(-1, 1)
            err_main = np.asarray(err_main).reshape(-1, 1)

            # Select best model
            best_metrics = _model_selector(x_main, y_main, err_main)
            best_model = best_metrics['model']
            best_poly_features = best_metrics['poly_features']
            
            # Get coefficients and intercept
            coefficients = best_model.coef_
            intercept = best_model.intercept_

            # Print comprehensive statistical report
            print("\n=== Polynomial Regression Results ===")
            print(f"Optimal degree: {best_metrics['degree']}")
            print(f"\nPolynomial Expression:")
            print(_construct_polynomial_expression(coefficients, intercept))
            
            print("\n=== Goodness-of-Fit Metrics ===")
            print(f"R² (Coefficient of Determination): {best_metrics['r2']:.4f}")
            print(f"Adjusted R²: {best_metrics['adj_r2']:.4f}")
            print(f"Akaike Information Criterion (AIC): {best_metrics['aic']:.2f}")
            print(f"Bayesian Information Criterion (BIC): {best_metrics['bic']:.2f}")
            
            print("\n=== Error Metrics ===")
            print(f"Mean Squared Error (MSE): {best_metrics['mse']:.4f}")
            print(f"Root Mean Squared Error (RMSE): {best_metrics['rmse']:.4f}")
            print(f"Mean Absolute Error (MAE): {best_metrics['mae']:.4f}")
            
            print("\n=== Residual Analysis ===")
            print(f"Residual Mean: {best_metrics['residual_stats']['mean']:.4f}")
            print(f"Residual Std Dev: {best_metrics['residual_stats']['std']:.4f}")
            print(f"Min Residual: {best_metrics['residual_stats']['min']:.4f}")
            print(f"Max Residual: {best_metrics['residual_stats']['max']:.4f}")

            return (
                lambda x_vals: best_model.predict(
                    best_poly_features.transform(np.array(x_vals).reshape(-1, 1))
                ).flatten(),
                best_metrics['degree']
            )

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
            raise RuntimeError(
                "Internal logic error: A function to generate predicted values is expected but not provided."
            )

        plt.figure()

        processed_data = self._read_main_data(main_data)
        title, x_label, y_label = processed_data[0], processed_data[1], processed_data[2]
        x_main, y_main, err_main = zip(*processed_data[3])

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if all(abs(err) < 1e-10 for err in err_main):
            plt.scatter(x_main, y_main, color="blue", label="Known Data")
        else:
            plt.errorbar(x_main, y_main, yerr=err_main, label="Known Data", fmt='o', capsize=5)

        if function:
            x_func = np.linspace(min(x_main), max(x_main), 100)
            y_func = function(list(x_func))
            plt.plot(x_func, y_func, color="green", label="Line of best fit")
            
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
