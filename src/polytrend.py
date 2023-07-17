import os
import random
from datetime import datetime
from typing import List, Tuple, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class PolyTrend:
    def _validate_data(self, degrees: List[int], known_data: List[Tuple[float, float]]) -> None:
        """
        Validate degrees and known_data to ensure they are not empty.

        Args:
            degrees (List[int]): List of polynomial degrees to consider.
            known_data (List[Tuple[float, float]]): List of tuples representing the known data points.

        Raises:
            ValueError: If degrees or known_data is empty.
        """
        if not degrees:
            raise ValueError("Degrees must be specified as a non-empty list.")

        if not known_data:
            raise ValueError("Known data must be specified as a non-empty list.")

    def polyplot(self, degrees: List[int], known_data: List[Tuple[float, float]], extrap_data: List[float] = None) -> None:
        """
        Plot the best polynomial fit on the known data.

        Args:
            degrees (List[int], optional): List of polynomial degrees to consider. Defaults to None.
            known_data (List[Tuple[float, float]]): List of tuples representing the known data points.
            extrap_data (List[float], optional): List of x coordinates for extrapolation. Defaults to None.

        Raises:
            ValueError: If degrees and/or known data is not specified or empty.

        Time Complexity: O(n * m * d), where n is the number of degrees, m is the number of known data points, and d is the maximum degree in degrees.
        Space Complexity: O(m), where m is the number of known data points.
        """
        self._validate_data(degrees, known_data)

        # find the best-fit polynomial function and plot the function and data
        self.polygraph(known_data, extrap_data, self.polyfind(degrees, known_data))

    def polyfind(self, degrees: List[int], known_data: List[Tuple[float, float]]) -> Callable[[List[float]], np.ndarray]:
        """
        Find the best-fit polynomial function.

        Args:
            degrees (List[int]): List of polynomial degrees to consider.
            known_data (List[Tuple[float, float]]): List of tuples representing the known data points.

        Returns:
            Callable[[List[float]], np.ndarray]: A function that predicts values based on the polynomial.

        Example:
            degrees = [1, 2, 3]
            known_data = [(1, 2), (2, 4), (3, 6)]
            poly_trend = PolyTrend()
            func = poly_trend.polyfind(degrees, known_data)
            output = func([4, 5])
            print(output)  # Output: array([ 8., 10.])

        Time Complexity: O(n * m * d), where n is the number of degrees, m is the number of known data points, and d is the maximum degree in degrees.
        Space Complexity: O(m), where m is the number of known data points.
        """
        self._validate_data(degrees, known_data)

        # shaping the data into usable structures
        x_known, y_known = zip(*known_data)

        # preallocate memory for best polynomial features and regression model
        best_poly_features = None
        best_reg = None

        # performance tracker (R-squared measure)
        best_score = float('-inf')

        # find the best-fit polynomial function; Brute Force approach
        for degree in degrees:
            # generate polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            x_poly = poly_features.fit_transform(np.array(x_known).reshape(-1, 1))

            # fit linear regression model
            reg = LinearRegression()
            reg.fit(x_poly, np.array(y_known).reshape(-1, 1))

            # calculate score for model evaluation
            score = reg.score(x_poly, np.array(y_known).reshape(-1, 1))

            # update the best score and models if a higher score is obtained
            if score > best_score:
                best_score = score
                best_poly_features = poly_features
                best_reg = reg

        # checking validity of model for output
        if best_reg is None or best_poly_features is None:
            raise ValueError("All degrees had a lower score than negative infinity. You might have discovered a new function...")
        else:
            print(f'Best R-squared score for given degree range: {best_score}')
            return lambda x_vals: best_reg.predict(best_poly_features.transform(np.array(x_vals).reshape(-1, 1)))

    def polygraph(self, known_data: List[Tuple[float, float]], extrap_data: List[float] = None, func: Optional[Callable[[List[float]], np.ndarray]] = None) -> None:
        """
        Plot the function, known data, and extrapolated data.

        Args:
            known_data (List[Tuple[float, float]]): List of tuples representing the known data points.
            extrap_data (List[float], optional): List of extrapolation data points. Defaults to None.
            func (Optional[Callable[[List[float]], np.ndarray]], optional): Function to generate predicted values. Defaults to None.

        Raises:
            ValueError: If known data is empty.
            ValueError: If extrapolation data is provided but no function is given.

        Time Complexity: O(k * (n + d)), where k is the number of extrapolation data points, n is the number of known data points,
        and d is the number of data points in extrap_data.
        Space Complexity: O(n + k + d), where n is the number of known data points, k is the number of extrapolation data points,
        and d is the number of data points in extrap_data.
        """
        self._validate_data([], known_data)

        # initializing graph prerequisites
        plt.figure()
        plt.title('Fitted Function via PolyTrend')
        plt.xlabel('x')
        plt.ylabel('f(x)')

        # plotting known data
        x_known, y_known = zip(*known_data)
        plt.scatter(x_known, y_known, label='Known data')

        # plotting extrapolated data if available
        if extrap_data is not None and func is not None:
            x_extrap = np.array(extrap_data)
            y_extrap = func(list(x_extrap))
            plt.scatter(x_extrap, y_extrap, color='red', label='Extrapolated data')

        # plotting the function
        x_func = np.linspace(min(x_known), max(x_known), 100)
        if func is not None:
            y_func = func(x_func)
            plt.plot(x_func, y_func, color='green', label='Fitted Function')

        plt.legend()

        # Add timestamp to the filename
        timestamp = datetime.now().strftime("%Y.%m-%d-%H:%M")
        filename = f'plot_{timestamp}_{random.randint(0,1000)}.png'

        # Save figure
        figures_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(figures_dir, exist_ok=True)
        filepath = os.path.join(figures_dir, filename)

        try:
            plt.savefig(filepath)
            print(f"Figure saved successfully: {filepath}")
        except Exception as e:
            print(f"Error saving figure: {str(e)}")

if __name__ == "__main__":
    # Example usage of PolyTrend class
    degrees = [1, 2, 3]
    known_data = [(1, 2), (2, 4), (3, 6)]
    poly_trend = PolyTrend()
    poly_trend.polyplot(degrees, known_data, extrap_data=[4, 5])
