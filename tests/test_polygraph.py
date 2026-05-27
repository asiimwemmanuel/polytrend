"""
tests/test_polygraph.py

Tests for PolyTrend.polygraph().

polygraph() is a rendering method: its primary output is a matplotlib figure,
not a return value. This makes Tier 2 (oracle) testing inapplicable — we
cannot cheaply assert pixel correctness.

What we *can* test:

  Tier 1 — Contract / control-flow invariants:
    - Must not raise on valid inputs.
    - Must raise RuntimeError when extrapolate_data is provided without a function.
    - Must call plt.show() exactly once per invocation (pinning the side-effect
      contract so a refactor that accidentally calls it zero or two times is caught).
    - Must call plt.scatter or plt.errorbar depending on whether errors are present.

  Tier 3 — Behavioural invariants via call inspection:
    - With zero errors, scatter (not errorbar) must be called.
    - With nonzero errors, errorbar (not scatter for the data) must be called.
    - Extrapolated points trigger an additional scatter call for the red markers.

All plt calls are intercepted via unittest.mock so no GUI is ever opened.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from polytrend import PolyTrend

FIXTURES = Path(__file__).parent / "fixtures"
pt = PolyTrend()

IDENTITY_FN = lambda xs: np.array(xs, dtype=float)
LINEAR_FN   = lambda xs: np.array([2 * x + 1 for x in xs], dtype=float)


def make_data(fn, xs, err=0.0):
    return [(float(x), float(fn([x])[0]), float(err)) for x in xs]


# ── autouse mock: suppress all matplotlib rendering ───────────────────────────

@pytest.fixture(autouse=True)
def mock_plt(monkeypatch):
    """Replace the pyplot module used inside polygraph with a MagicMock."""
    mock = MagicMock()
    monkeypatch.setattr("polytrend.plt", mock)
    return mock


# ── Tier 1: contract invariants ───────────────────────────────────────────────

class TestPolygraphContract:
    def test_no_raise_on_minimal_valid_input(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 6))
        pt.polygraph(data)  # no function, no extrapolation

    def test_no_raise_with_function(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 11))
        pt.polygraph(data, function=LINEAR_FN)

    def test_no_raise_with_extrapolation(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 11))
        pt.polygraph(data, extrapolate_data=[15.0, 20.0], function=LINEAR_FN)

    def test_raises_runtime_error_extrapolate_without_function(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 11))
        with pytest.raises(RuntimeError):
            pt.polygraph(data, extrapolate_data=[15.0])

    def test_show_called_exactly_once(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 6))
        pt.polygraph(data)
        mock_plt.show.assert_called_once()

    def test_figure_created(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 6))
        pt.polygraph(data)
        mock_plt.figure.assert_called_once()

    def test_grid_enabled(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 6))
        pt.polygraph(data)
        mock_plt.grid.assert_called_once_with(True)

    def test_legend_called(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 6))
        pt.polygraph(data)
        mock_plt.legend.assert_called_once()

    def test_csv_path_accepted(self, mock_plt):
        """polygraph must accept a CSV path and not raise."""
        pt.polygraph(str(FIXTURES / "linear_clean.csv"), function=LINEAR_FN)

    def test_accepts_preprocessed_data(self, mock_plt):
        """polygraph must accept pre-processed output from _read_main_data."""
        processed = pt._read_main_data(make_data(LINEAR_FN, range(1, 6)))
        pt.polygraph(processed)


# ── Tier 3: behavioural / rendering branch invariants ────────────────────────

class TestPolygraphBranches:
    def test_zero_errors_uses_scatter_not_errorbar(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 11), err=0.0)
        pt.polygraph(data)
        mock_plt.scatter.assert_called()
        mock_plt.errorbar.assert_not_called()

    def test_nonzero_errors_uses_errorbar_not_scatter_for_data(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 11), err=1.0)
        pt.polygraph(data)
        mock_plt.errorbar.assert_called()

    def test_function_triggers_plot_call(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 11))
        pt.polygraph(data, function=LINEAR_FN)
        mock_plt.plot.assert_called()

    def test_no_function_no_plot_call(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 11))
        pt.polygraph(data, function=None)
        mock_plt.plot.assert_not_called()

    def test_extrapolation_adds_scatter_call(self, mock_plt):
        """
        With extrapolation, scatter must be called at least twice:
        once for known data (zero errors), once for extrapolated points.
        """
        data = make_data(LINEAR_FN, range(1, 11), err=0.0)
        pt.polygraph(data, extrapolate_data=[15.0, 20.0], function=LINEAR_FN)
        assert mock_plt.scatter.call_count >= 2

    def test_extrapolation_adds_second_plot_call(self, mock_plt):
        """
        The dashed extension segment requires a second plt.plot call beyond
        the first curve call.
        """
        data = make_data(LINEAR_FN, range(1, 11))
        pt.polygraph(data, extrapolate_data=[15.0, 20.0], function=LINEAR_FN)
        assert mock_plt.plot.call_count >= 2

    def test_annotate_called_once_per_extrapolated_point(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 11))
        extrap = [15.0, 20.0, 25.0]
        pt.polygraph(data, extrapolate_data=extrap, function=LINEAR_FN)
        assert mock_plt.annotate.call_count == len(extrap)

    def test_xlabel_and_ylabel_set(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 6))
        pt.polygraph(data)
        mock_plt.xlabel.assert_called_once()
        mock_plt.ylabel.assert_called_once()

    def test_title_set(self, mock_plt):
        data = make_data(LINEAR_FN, range(1, 6))
        pt.polygraph(data)
        mock_plt.title.assert_called_once()
