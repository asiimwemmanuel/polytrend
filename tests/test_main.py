import tests.test_main as test_main
# import numpy as np
from unittest.mock import patch
# from io import StringIO
import pandas as pd

from polytrend import PolyTrend

class TestPolyTrend:
    def test_polyfind_returns_callable(self):
        polytrend_instance = PolyTrend()
        degrees = [1, 2, 3]
        main_data = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
        result = polytrend_instance.polyfind(degrees, main_data)
        assert callable(result), "The result should be a callable function"

    def test_polygraph_raises_error_without_function(self):
        polytrend_instance = PolyTrend()
        main_data = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
        extrapolate_data = [4.0, 5.0]
        with test_main.raises(ValueError):
            polytrend_instance.polygraph(main_data, extrapolate_data)

    @patch("pandas.read_csv")
    def test_read_main_data_handles_input_types(self, mock_read_csv):
        # Mocking pandas.read_csv to return a specific DataFrame
        mock_read_csv.return_value = pd.DataFrame({
            'x': [1, 2, 3],
            'f(x)': [2, 3, 5]
        })
        polytrend_instance = PolyTrend()

        # Testing with list of tuples
        main_data_list = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
        result_list = polytrend_instance._read_main_data(main_data_list)
        assert isinstance(result_list, list), "Should return a list"
        assert result_list[3] == main_data_list, "Should correctly extract data from list of tuples"

        # Testing with CSV file path
        main_data_csv = "path/to/data.csv"
        result_csv = polytrend_instance._read_main_data(main_data_csv)
        assert isinstance(result_csv, list), "Should return a list"
        assert result_csv[3] == [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)], "Should correctly extract data from CSV file"

    @test_main.mark.parametrize("degrees, main_data, expected_degree", [
        ([1, 2, 3], [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)], 2),
        ([2, 3, 4], [(1.0, 1.0), (2.0, 4.0), (3.0, 9.0)], 3),
    ])
    def test_polyfind_with_various_degrees(degrees, main_data, expected_degree):
        polytrend_instance = PolyTrend()
        result_degree = polytrend_instance.polyfind(degrees, main_data)[1]
        assert result_degree == expected_degree, "The polynomial degree should match the expected"
