"""
tests/test_read_main_data.py

Tier 1 — structural / contract invariants for PolyTrend._read_main_data.

These tests make no claims about numeric accuracy; they only verify that
the method returns data in the shape and type the rest of the class expects.
They also pin the two distinct code paths (list input vs CSV input) so that
drift between them becomes visible immediately.
"""

import pytest
from pathlib import Path
from polytrend import PolyTrend

FIXTURES = Path(__file__).parent / 'fixtures'
pt = PolyTrend()


# ── helpers ──────────────────────────────────────────────────────────────────


def assert_processed_shape(result, expected_n: int):
  """Assert the canonical [title, x_label, y_label, data_points] contract."""
  assert isinstance(result, list), 'result must be a list'
  assert len(result) == 4, 'result must have exactly 4 elements'

  title, x_label, y_label, data_points = result

  assert isinstance(title, str) and title, 'title must be a non-empty string'
  assert isinstance(x_label, str) and x_label, 'x_label must be a non-empty string'
  assert isinstance(y_label, str) and y_label, 'y_label must be a non-empty string'
  assert isinstance(data_points, list), 'data_points must be a list'
  assert len(data_points) == expected_n, (
    f'expected {expected_n} points, got {len(data_points)}'
  )

  for point in data_points:
    assert isinstance(point, tuple), 'each data point must be a tuple'
    assert len(point) == 3, 'each data point must have exactly 3 elements (x, y, err)'
    x, y, err = point
    assert isinstance(x, float), f'x must be float, got {type(x)}'
    assert isinstance(y, float), f'y must be float, got {type(y)}'
    assert isinstance(err, float), f'err must be float, got {type(err)}'


# ── list-input path ───────────────────────────────────────────────────────────


class TestListInput:
  def test_minimal_valid_input(self):
    data = [(1.0, 2.0, 0.0), (3.0, 4.0, 0.0)]
    result = pt._read_main_data(data)
    assert_processed_shape(result, expected_n=2)

  def test_values_are_preserved(self):
    data = [(1.0, 3.0, 0.5), (2.0, 5.0, 0.5)]
    _, _, _, points = pt._read_main_data(data)
    assert points[0] == (1.0, 3.0, 0.5)
    assert points[1] == (2.0, 5.0, 0.5)

  def test_integer_inputs_cast_to_float(self):
    """Integers should be silently promoted — the class promises floats."""
    data = [(1, 2, 0), (3, 4, 0)]
    _, _, _, points = pt._read_main_data(data)
    for x, y, err in points:
      assert isinstance(x, float)
      assert isinstance(y, float)
      assert isinstance(err, float)

  def test_negative_values_pass_through(self):
    data = [(-5.0, -10.0, 0.0), (-1.0, -2.0, 0.0)]
    _, _, _, points = pt._read_main_data(data)
    assert points[0][0] == -5.0
    assert points[0][1] == -10.0

  def test_generic_labels_used_for_list_input(self):
    """List path must use generic 'x' / 'f(x)' labels, not custom ones."""
    data = [(1.0, 2.0, 0.0), (3.0, 6.0, 0.0)]
    _, x_label, y_label, _ = pt._read_main_data(data)
    assert x_label == 'x'
    assert y_label == 'f(x)'

  def test_large_input(self):
    data = [(float(i), float(i**2), 0.0) for i in range(200)]
    result = pt._read_main_data(data)
    assert_processed_shape(result, expected_n=200)

  def test_invalid_type_raises_value_error(self):
    with pytest.raises((ValueError, TypeError)):
      pt._read_main_data(42)

  def test_none_raises(self):
    with pytest.raises((ValueError, TypeError, AttributeError)):
      pt._read_main_data(None)


# ── CSV-input path ────────────────────────────────────────────────────────────


class TestCSVInput:
  def test_two_column_csv(self):
    result = pt._read_main_data(str(FIXTURES / 'linear_clean.csv'))
    assert_processed_shape(result, expected_n=20)

  def test_three_column_csv_preserves_errors(self):
    """CSV with an error column should produce non-zero err values."""
    _, _, _, points = pt._read_main_data(str(FIXTURES / 'uniform_errors.csv'))
    errors = [err for _, _, err in points]
    assert all(err == 1.0 for err in errors), (
      'uniform error of 1.0 should be read from CSV'
    )

  def test_csv_labels_come_from_header(self):
    """CSV path must use the header row for axis labels."""
    _, x_label, y_label, _ = pt._read_main_data(str(FIXTURES / 'linear_clean.csv'))
    assert x_label == 'x'
    assert y_label == 'f(x)'

  def test_csv_title_contains_filename(self):
    path = str(FIXTURES / 'linear_clean.csv')
    title, _, _, _ = pt._read_main_data(path)
    assert 'linear_clean.csv' in title

  def test_missing_file_raises(self):
    with pytest.raises((FileNotFoundError, OSError)):
      pt._read_main_data('/nonexistent/path/fake.csv')

  def test_two_point_csv(self):
    """n=2 is the minimum; should not raise."""
    result = pt._read_main_data(str(FIXTURES / 'two_points.csv'))
    assert_processed_shape(result, expected_n=2)

  def test_large_csv(self):
    result = pt._read_main_data(str(FIXTURES / 'large.csv'))
    assert_processed_shape(result, expected_n=200)

  def test_negative_x_csv(self):
    _, _, _, points = pt._read_main_data(str(FIXTURES / 'negative_x.csv'))
    xs = [x for x, _, _ in points]
    assert any(x < 0 for x in xs), 'negative x values should survive CSV round-trip'

  def test_missing_error_column_defaults_to_zero(self):
    """
    When the CSV has no error column, err must default to 0.0 (float).
    This pins the documented behaviour: 'CSV path does not currently support
    error column; default to 0.'  The value being int(0) vs float(0.0) is
    an existing inconsistency — this test will flag if it ever changes.
    """
    _, _, _, points = pt._read_main_data(str(FIXTURES / 'linear_clean.csv'))
    for _, _, err in points:
      assert float(err) == 0.0


# ── path-equivalence ──────────────────────────────────────────────────────────


class TestPathEquivalence:
  """
  Both input paths must produce data_points with the same numeric content
  when given the same underlying numbers.  Labels and title may differ.
  """

  def test_list_and_csv_agree_on_xy_values(self):
    # linear_clean.csv contains y = 2x + 1 for x in 1..20
    csv_result = pt._read_main_data(str(FIXTURES / 'linear_clean.csv'))
    list_data = [(float(x), float(2 * x + 1), 0.0) for x in range(1, 21)]
    list_result = pt._read_main_data(list_data)

    csv_points = csv_result[3]
    list_points = list_result[3]

    assert len(csv_points) == len(list_points)
    for (cx, cy, _), (lx, ly, _) in zip(csv_points, list_points):
      assert abs(cx - lx) < 1e-9
      assert abs(cy - ly) < 1e-9
