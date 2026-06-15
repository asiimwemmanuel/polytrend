"""
tests/test_polyfind.py

Tests for PolyTrend.polyfind().

Three tiers are covered:

  Tier 1 — Contract invariants:
    The method must always return (callable, int). The callable must accept
    a list of floats and return a numpy array of the same length.

  Tier 2 — Oracle / algebraic invariants:
    When data is generated from a known polynomial with no noise, the fitted
    function must reproduce training-point values to within numerical tolerance.
    We also assert that BIC selects the correct degree when the true degree is
    included in the candidate list and data is noise-free.

  Tier 3 — Behavioural / comparative invariants:
    - Noise increases MSE relative to a noise-free fit on the same polynomial.
    - Higher candidates should not blindly win; BIC must penalise them on
      low-degree data.
    - The weighted (error) path and unweighted path both return valid callables.
    - BIC degree selection: on clearly linear data, degree-1 should win over
      degree-5 when both are candidates.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from polytrend import PolyTrend

FIXTURES = Path(__file__).parent / 'fixtures'
pt = PolyTrend()


# suppress plt.show() across all tests in this file
@pytest.fixture(autouse=True)
def no_show(monkeypatch):
  monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)


# ── helpers ───────────────────────────────────────────────────────────────────


def make_data(fn, xs, err=0.0):
  """Build a list-of-tuples dataset from a ground-truth function."""
  return [(float(x), float(fn(x)), float(err)) for x in xs]


def mse(fn, data):
  """Mean squared error of fn on (x, y) pairs."""
  return np.mean([(fn([x]) - y) ** 2 for x, y, _ in data])


# ── Tier 1: contract invariants ───────────────────────────────────────────────


class TestPolyfindContract:
  def test_returns_tuple_of_callable_and_int(self):
    data = make_data(lambda x: 2 * x + 1, range(1, 11))
    result = pt.polyfind([1, 2], data)
    assert isinstance(result, tuple) and len(result) == 2
    fn, degree = result
    assert callable(fn)
    assert isinstance(degree, int)

  def test_callable_accepts_list_returns_ndarray(self):
    data = make_data(lambda x: 2 * x + 1, range(1, 11))
    fn, _ = pt.polyfind([1], data)
    out = fn([5.0, 10.0, 15.0])
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)

  def test_callable_length_matches_input_length(self):
    data = make_data(lambda x: x**2, range(1, 11))
    fn, _ = pt.polyfind([1, 2], data)
    for n in [1, 5, 50]:
      assert fn(list(range(n))).shape == (n,)

  def test_degree_is_from_candidate_list(self):
    data = make_data(lambda x: x**2 - x + 1, range(1, 16))
    _, degree = pt.polyfind([1, 2, 3], data)
    assert degree in [1, 2, 3]

  def test_single_candidate_degree_is_used(self):
    """When only one degree is offered, it must be selected."""
    data = make_data(lambda x: 2 * x + 1, range(1, 11))
    _, degree = pt.polyfind([3], data)
    assert degree == 3

  def test_csv_path_accepted(self):
    fn, degree = pt.polyfind([1, 2, 3], str(FIXTURES / 'linear_clean.csv'))
    assert callable(fn)
    assert isinstance(degree, int)

  def test_empty_degrees_list_raises(self):
    data = make_data(lambda x: x, range(1, 5))
    with pytest.raises((ValueError, IndexError, Exception)):
      pt.polyfind([], data)


# ── Tier 2: oracle / algebraic invariants ────────────────────────────────────


class TestPolyfindOracle:
  """
  For noise-free data generated from a known polynomial of degree d,
  the fitted function must reproduce training values to within 1e-4.
  Exact degree recovery is also asserted when d is in the candidate list
  and data is clean, because BIC will assign -inf (perfect fit) to it.
  """

  def _assert_reconstruction(self, fn, data, tol=1e-4):
    for x, y_true, _ in data:
      y_pred = fn([x])[0]
      assert abs(y_pred - y_true) < tol, (
        f'At x={x}: predicted {y_pred:.6f}, expected {y_true:.6f} '
        f'(diff={abs(y_pred - y_true):.2e})'
      )

  def test_linear_clean_reconstructs(self):
    data = make_data(lambda x: 2 * x + 1, range(1, 21))
    fn, _ = pt.polyfind([1, 2, 3], data)
    self._assert_reconstruction(fn, data)

  def test_quadratic_clean_reconstructs(self):
    data = make_data(lambda x: x**2 - 3 * x + 2, range(1, 21))
    fn, _ = pt.polyfind([1, 2, 3], data)
    self._assert_reconstruction(fn, data)

  def test_cubic_clean_reconstructs(self):
    data = make_data(lambda x: x**3 - x**2 + x - 1, range(1, 21))
    fn, _ = pt.polyfind([1, 2, 3, 4], data)
    self._assert_reconstruction(fn, data)

  def test_linear_degree_selected_on_clean_linear_data(self):
    """BIC must select degree 1 on exact linear data."""
    data = make_data(lambda x: 2 * x + 1, range(1, 21))
    _, degree = pt.polyfind([1, 2, 3], data)
    assert degree == 1, f'Expected degree 1 on clean linear data, got {degree}'

  def test_quadratic_degree_selected_on_clean_quadratic_data(self):
    data = make_data(lambda x: x**2 - 3 * x + 2, range(1, 21))
    _, degree = pt.polyfind([1, 2, 3], data)
    assert degree == 2, f'Expected degree 2 on clean quadratic data, got {degree}'

  def test_cubic_degree_selected_on_clean_cubic_data(self):
    data = make_data(lambda x: x**3 - x**2 + x - 1, range(1, 21))
    _, degree = pt.polyfind([1, 2, 3], data)
    assert degree == 3, f'Expected degree 3 on clean cubic data, got {degree}'

  def test_two_point_minimum(self):
    """n=2 is the minimum; a degree-1 fit must succeed."""
    data = [(1.0, 3.0, 0.0), (2.0, 5.0, 0.0)]  # y = 2x + 1
    fn, _ = pt.polyfind([1], data)
    assert abs(fn([1.0])[0] - 3.0) < 1e-4
    assert abs(fn([2.0])[0] - 5.0) < 1e-4

  def test_negative_x_values(self):
    data = make_data(lambda x: 2 * x + 1, range(-10, 11))
    fn, _ = pt.polyfind([1, 2], data)
    self._assert_reconstruction(fn, data)

  def test_csv_linear_clean_reconstructs(self):
    fn, _ = pt.polyfind([1, 2, 3], str(FIXTURES / 'linear_clean.csv'))
    # linear_clean: y = 2x + 1 for x in 1..20
    for x in range(1, 21):
      assert abs(fn([float(x)])[0] - (2 * x + 1)) < 1e-4


# ── Tier 3: behavioural / comparative invariants ──────────────────────────────


class TestPolyfindBehavioural:
  def test_noise_increases_mse(self):
    """
    A fit on noisy data should have higher MSE (evaluated on the training
    set) than a fit on the same polynomial without noise.
    """
    np.random.seed(0)
    xs = list(range(1, 31))
    clean = make_data(lambda x: x**2 - 3 * x + 2, xs)
    noisy = [(x, y + np.random.normal(0, 10), e) for x, y, e in clean]

    fn_clean, _ = pt.polyfind([1, 2, 3], clean)
    fn_noisy, _ = pt.polyfind([1, 2, 3], noisy)

    mse_clean = mse(fn_clean, clean)
    mse_noisy = mse(fn_noisy, noisy)

    assert mse_clean < mse_noisy, (
      f'Clean MSE ({mse_clean:.4f}) should be less than noisy MSE ({mse_noisy:.4f})'
    )

  def test_bic_penalises_high_degree_on_linear_data(self):
    """
    On clearly linear data, BIC must not blindly select the highest degree.
    Degree 1 should win over degree 5.
    """
    data = make_data(lambda x: 3 * x - 7, range(1, 25))
    _, degree = pt.polyfind([1, 2, 3, 4, 5], data)
    assert degree == 1, (
      f'BIC should select degree 1 on clean linear data, not degree {degree}'
    )

  def test_weighted_path_returns_valid_callable(self):
    """Non-zero errors trigger Ridge regression; result must still be callable."""
    data = make_data(lambda x: 2 * x + 1, range(1, 21), err=1.0)
    fn, _ = pt.polyfind([1, 2], data)
    out = fn([5.0, 10.0])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)

  def test_mixed_errors_triggers_weighted_path(self):
    """
    A single nonzero error in an otherwise zero array flips to Ridge for
    the whole fit (np.any(errors != 0) is True).  Zero-error points are
    clamped to the smallest nonzero error before inversion so that no
    divide-by-zero occurs and those points receive a disproportionately
    large (but finite) weight.
    """
    data = make_data(lambda x: 2 * x + 1, range(1, 21))
    # inject one nonzero error — zeros remain for all other points
    data[5] = (data[5][0], data[5][1], 2.0)
    fn, _ = pt.polyfind([1, 2], data)
    out = fn([1.0, 10.0, 20.0])
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    # the fit should still be in a plausible range for clean linear data
    for x, y, _ in data:
      assert abs(fn([x])[0] - y) < 5.0, f'Prediction unexpectedly far off at x={x}'

  def test_uniform_errors_csv(self):
    """CSV with error column should use Ridge and return a valid function."""
    fn, _ = pt.polyfind([1, 2, 3], str(FIXTURES / 'uniform_errors.csv'))
    out = fn([1.0, 10.0, 20.0])
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)

  def test_large_dataset_completes(self):
    """n=200 should not raise or time out (basic performance pin)."""
    fn, degree = pt.polyfind([1, 2, 3], str(FIXTURES / 'large.csv'))
    assert callable(fn)
    assert degree in [1, 2, 3]

  def test_duplicate_x_values_do_not_raise(self):
    """sklearn handles duplicate x; polyfind must not crash."""
    fn, _ = pt.polyfind([1, 2], str(FIXTURES / 'duplicate_x.csv'))
    assert callable(fn)

  def test_constant_data_does_not_raise(self):
    """y=constant is a degenerate signal; polyfind must not crash."""
    fn, _ = pt.polyfind([1, 2, 3], str(FIXTURES / 'constant.csv'))
    assert callable(fn)
