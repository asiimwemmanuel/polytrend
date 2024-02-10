# strictly for research purposes, out of commission

import os
import random
from typing import List, Tuple, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class PolyTrend:
    def __init__(self) -> None:
        pass

    def polyplot(self, known_data: List[Tuple[float, float]], extrap_data: List[float] = [], degrees: List[int] = None) -> None:
        """
        Plot the best polynomial fit on the known data.

        Args:
            known_data (List[Tuple[float, float]]): List of tuples representing the known data points.
            extrap_data (List[float], optional): List of extrapolation data points. Defaults to [].
            degrees (List[int], optional): List of polynomial degrees to consider. Defaults to None.

        Raises:
            ValueError: If degrees is not specified or empty.
        """
        if degrees is None or len(degrees) == 0:
            raise ValueError("Degrees must be specified as a non-empty list.")

        func = self.polyfind(known_data, degrees)

        self.graph(known_data, extrap_data, func)

    def polyfind(self, known_data: List[Tuple[float, float]], degrees: List[int]) -> Callable[[List[float]], np.ndarray]:
        """
        Find the best-fit polynomial function.

        Args:
            known_data (List[Tuple[float, float]]): List of tuples representing the known data points.
            degrees (List[int]): List of polynomial degrees to consider.

        Returns:
            Callable[[List[float]], np.ndarray]: Lambda function for the predicted values based on the polynomial.
        """
        x_known, y_known = np.asarray(known_data).T

        best_deg = -1
        best_score = float('-inf')

        # Iterate over the specified degrees and find the best degree polynomial
        for degree in degrees:
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            x_poly = poly_features.fit_transform(x_known.reshape(-1, 1))

            reg = LinearRegression()
            reg.fit(x_poly, y_known)

            score = reg.score(x_poly, y_known)

            # Update the best degree and score if a higher score is obtained
            if score > best_score:
                best_score = score
                best_deg = degree

        # Fit the best degree polynomial
        poly_features = PolynomialFeatures(degree=best_deg, include_bias=False)
        x_poly = poly_features.fit_transform(x_known.reshape(-1, 1))
        reg = LinearRegression()
        reg.fit(x_poly, y_known)

        return lambda x_vals: np.array(reg.predict(poly_features.transform(np.array(x_vals).reshape(-1, 1))))

    def graph(self, known_data: List[Tuple[float, float]], extrap_data: List[float] = [], func: Optional[Callable[[List[float]], np.ndarray]] = None) -> None:
        """
        Plot the function, known data, and extrapolated data.

        Args:
            known_data (List[Tuple[float, float]]): List of tuples representing the known data points.
            extrap_data (List[float], optional): List of extrapolation data points. Defaults to [].
            func (Optional[Callable[[List[float]], np.ndarray]], optional): Function to generate predicted values. Defaults to None.

        Raises:
            ValueError: If extrapolation data is provided but no function is given.
        """
        x_known, y_known = np.asarray(known_data).T

        plt.figure()
        plt.title('Fitted Function via PolyTrend')
        plt.xlabel('x')
        plt.ylabel('f(x)')

        # Plot known data
        plt.scatter(x_known, y_known, color='blue', label='Known data')

        if extrap_data:
            if func is None:
                raise ValueError("Extrapolation data is provided, but no function is given.")

            x_extrap = np.array(extrap_data)
            y_extrap = func(list(x_extrap))
            plt.scatter(x_extrap, y_extrap, color='red', label='Extrapolated data')

            x_func = np.linspace(min(np.concatenate((x_known, x_extrap))), max(np.concatenate((x_known, x_extrap))), 100)
            y_func = func(x_func)
            plt.plot(x_func, y_func, color='green', label='Fitted Function')

        if func is not None:
            x_func = np.linspace(min(x_known), max(x_known), 100)
            y_func = func(x_func)
            plt.plot(x_func, y_func, color='green', label='Fitted Function')

        plt.legend()
        # plt.show()

        # Comment out or modify the saving part based on your requirements
        filename = f'plot_{random.uniform(0, 1)}.png'
        figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        filepath = os.path.join(figures_dir, filename)
        plt.savefig(filepath)

    def findnth(self, seq: list, n: int):
        pass
