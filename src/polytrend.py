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

from csv import reader
from random import uniform
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


class PolyTrend:
  """
  PolyTrend: A utility for polynomial trend fitting, visualization, and extrapolation.

  Main Methods:
    - polyplot(): Finds and plots the best polynomial fit on the known data.
    - polyfind(): Finds the best-fit polynomial function.
    - polygraph(): Plots the function, known data, and extrapolated data.
  """

  def _read_main_data(
    self, main_data: Union[list[Tuple[float, float, float]], str]
  ) -> list:
    """
    Read and extract known data from either a list of tuples or a CSV file.

    Args:
      main_data: Known data points as list of (x, y, err) tuples, or a CSV file path.

    Returns:
      [graph_title, x_label, y_label, list[Tuple[float, float, float]]]
    """
    if isinstance(main_data, str):
      with open(main_data, mode="r") as file:
        csv = reader(file)
        headers = next(csv)
        x_label = headers[0].strip()
        y_label = headers[1].strip()
        rows = list(csv)

      x_vals = [r[0] for r in rows]
      y_vals = [r[1] for r in rows]
      # CSV path does not currently support error column; default to 0
      err_vals = [r[2] if len(r) >= 3 else 0 for r in rows]
      title = f"Fitted Function via PolyTrend - Data from {main_data}"

    elif isinstance(main_data, list):
      x_label = "x"
      y_label = "f(x)"
      x_vals, y_vals, err_vals = zip(*main_data)
      title = "Fitted Function via PolyTrend"

    else:
      raise ValueError("main_data must be a list of (x, y, err) tuples or a CSV file path")

    data_points = [
      (float(x), float(y), float(e))
      for x, y, e in zip(x_vals, y_vals, err_vals)
    ]

    return [title, x_label, y_label, data_points]

  def polyplot(
    self,
    degrees: list[int],
    main_data: Union[list[Tuple[float, float, float]], str],
    extrapolate_data: Optional[list[float]] = None,
  ) -> None:
    """
    Find and plot the best polynomial fit on the known data.

    Args:
      degrees: Polynomial degrees to evaluate.
      main_data: Known data as list of (x, y, err) tuples or CSV path.
      extrapolate_data: x-coordinates to extrapolate. Defaults to None.
    """
    extrapolate_data = extrapolate_data or []
    processed = self._read_main_data(main_data)
    fitted_fn, _ = self.polyfind(degrees, processed)
    self.polygraph(processed, extrapolate_data, function=fitted_fn)

  def polyfind(
    self,
    degrees: list[int],
    main_data: Union[list[Tuple[float, float, float]], str, list],
  ) -> Tuple[Callable[[list], np.ndarray], int]:
    """
    Find the best-fit polynomial, selected by lowest BIC across candidate degrees.

    Args:
      degrees: Polynomial degrees to evaluate.
      main_data: Known data as list of (x, y, err) tuples, CSV path, or
                 pre-processed output from _read_main_data.

    Returns:
      (predict_fn, best_degree) where predict_fn maps x-values to predictions.
      Prints a statistical report to stdout.
    """

    def _polynomial_str(coefficients: np.ndarray, intercept: np.ndarray) -> str:
      """Build a human-readable polynomial expression string."""
      coeffs = np.asarray(coefficients, dtype=float).ravel()[::-1]
      intercept_val: float = float(np.asarray(intercept).ravel()[0])
      n_coeffs = len(coeffs)
      parts: list[str] = []

      for i, coeff in enumerate(coeffs):
        if coeff == 0:
          continue
        rounded = round(coeff, 3)
        power = n_coeffs - i
        x_term = "x" if power == 1 else f"x^{power}"

        if not parts:
          parts.append(f"({rounded}){x_term}")
        else:
          sign = "+ " if rounded > 0 else "- "
          parts.append(f"{sign}({abs(rounded)}){x_term}")

      intercept_rounded = round(intercept_val, 3)
      if intercept_rounded != 0:
        sign = "+ " if intercept_rounded > 0 else "- "
        parts.append(f"{sign}{abs(intercept_rounded)}")

      return " ".join(parts)

    def _metrics(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> dict:
      """Compute fit quality metrics for a candidate model."""
      n = len(y_true)
      if n == 0:
        raise ValueError("y_true and y_pred must not be empty")

      residuals = y_true - y_pred
      mse = mean_squared_error(y_true, y_pred)
      r2 = r2_score(y_true, y_pred)

      adj_r2 = (
        1 - (1 - r2) * (n - 1) / (n - n_params - 1)
        if n > n_params + 1 and np.isfinite(r2)
        else np.nan
      )

      # mse == 0 is a perfect fit: assign -inf so it always wins the BIC comparison
      if mse == 0:
        aic = bic = -float("inf")
      elif n > 0:
        aic = n * np.log(mse) + 2 * n_params
        bic = n * np.log(mse) + n_params * np.log(n)
      else:
        aic = bic = np.nan

      return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2,
        "adj_r2": adj_r2,
        "aic": aic,
        "bic": bic,
        "residual_stats": {
          "mean": np.mean(residuals),
          "std": np.std(residuals, ddof=0),
          "min": np.min(residuals),
          "max": np.max(residuals),
        },
      }

    def _select_model(
      x: np.ndarray, y: np.ndarray, errors: np.ndarray
    ) -> dict:
      """Fit each candidate degree and return metrics for the best (lowest BIC)."""
      weighted = np.any(errors != 0)
      best: dict = {"bic": float("inf"), "model": None}

      for degree in degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly = poly.fit_transform(x)

        if weighted:
          reg = Ridge(alpha=1.0)
          reg.fit(x_poly, y, sample_weight=1 / errors.ravel())
        else:
          reg = LinearRegression()
          reg.fit(x_poly, y)

        y_pred = reg.predict(x_poly)
        candidate = _metrics(y, y_pred, n_params=x_poly.shape[1])
        candidate.update({"model": reg, "poly_features": poly, "degree": degree})

        if candidate["bic"] < best["bic"]:
          best = candidate

      if best["model"] is None:
        raise ValueError("No valid model found across candidate degrees.")

      return best

    # Accept either raw data or pre-processed output from _read_main_data
    if isinstance(main_data, list) and main_data and isinstance(main_data[0], str):
      processed = main_data  # already processed
    else:
      processed = self._read_main_data(main_data)

    x, y, err = zip(*processed[3])
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    err = np.asarray(err).reshape(-1, 1)

    best = _select_model(x, y, err)
    coef = best["model"].coef_
    intercept = best["model"].intercept_

    print("\n=== Polynomial Regression Results ===")
    print(f'Optimal degree: {best["degree"]}')
    print("\nPolynomial Expression:")
    print(_polynomial_str(coef, intercept))

    print("\n=== Goodness-of-Fit Metrics ===")
    print(f'R²: {best["r2"]:.4f}')
    print(f'Adjusted R²: {best["adj_r2"]:.4f}')
    print(f'AIC: {best["aic"]:.2f}')
    print(f'BIC: {best["bic"]:.2f}')

    print("\n=== Error Metrics ===")
    print(f'MSE:  {best["mse"]:.4f}')
    print(f'RMSE: {best["rmse"]:.4f}')
    print(f'MAE:  {best["mae"]:.4f}')

    print("\n=== Residual Analysis ===")
    print(f'Mean: {best["residual_stats"]["mean"]:.4f}')
    print(f'Std:  {best["residual_stats"]["std"]:.4f}')
    print(f'Min:  {best["residual_stats"]["min"]:.4f}')
    print(f'Max:  {best["residual_stats"]["max"]:.4f}')

    predict_fn = lambda x_vals: best["model"].predict(
      best["poly_features"].transform(np.array(x_vals).reshape(-1, 1))
    ).flatten()

    return predict_fn, best["degree"]

  def polygraph(
    self,
    main_data: Union[list[Tuple[float, float, float]], str, list],
    extrapolate_data: Optional[list[float]] = None,
    function: Optional[Callable[[list[float]], np.ndarray]] = None,
  ) -> None:
    """
    Plot the polynomial function against known and extrapolated data.

    Args:
      main_data: Known data as list of (x, y, err) tuples, CSV path, or
                 pre-processed output from _read_main_data.
      extrapolate_data: x-coordinates to extrapolate. Defaults to None.
      function: Prediction function from polyfind(). Defaults to None.

    Raises:
      RuntimeError: If extrapolate_data is provided without a function.
    """
    extrapolate_data = extrapolate_data or []

    if extrapolate_data and function is None:
      raise RuntimeError(
        "Internal logic error: extrapolate_data provided but no prediction function given."
      )

    # Accept pre-processed data to avoid redundant parsing when called from polyplot
    if isinstance(main_data, list) and main_data and isinstance(main_data[0], str):
      processed = main_data
    else:
      processed = self._read_main_data(main_data)

    title, x_label, y_label = processed[0], processed[1], processed[2]
    x_main, y_main, err_main = zip(*processed[3])

    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if all(abs(e) < 1e-10 for e in err_main):
      plt.scatter(x_main, y_main, color="blue", label="Known Data")
    else:
      plt.errorbar(x_main, y_main, yerr=err_main, label="Known Data", fmt="o", capsize=5)

    if function is not None:
      x_curve = np.linspace(min(x_main), max(x_main), 100)
      plt.plot(x_curve, function(list(x_curve)), color="green", label="Line of best fit")

      if extrapolate_data:
        y_extrap = function(list(extrapolate_data))
        x_ext = np.linspace(max(x_main), max(extrapolate_data), 100)
        plt.plot(x_ext, function(list(x_ext)), color="green")
        plt.scatter(extrapolate_data, y_extrap, color="red", label="Extrapolated data")

        for x, y in zip(extrapolate_data, y_extrap):
          plt.annotate(
            f"({x:.2f}, {y:.2f})",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=0.5),
          )

    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
  degrees = [1, 2, 3]
  data = [
    (float(x), float(0.5 * x**2 - 2 * x + 1 + uniform(-1000, 1000)), 1.0)
    for x in range(0, 100)
  ]
  try:
    PolyTrend().polyplot(degrees, data, extrapolate_data=[15, 20])
  except ValueError as e:
    print(f"Error: {e}")
  print("Example usage completed.")
