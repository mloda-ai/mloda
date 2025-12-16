from mloda.user import ParallelizationMode
import pytest
from mloda.user import FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class TestPythonDictFramework:
    """Test suite for PythonDict compute framework."""

    def test_expected_data_framework(self) -> None:
        assert PythonDictFramework.expected_data_framework() == list

    def test_transform_dict_to_list(self) -> None:
        """Test transformation from columnar dict to list of dicts."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        # Test columnar dict transformation
        input_data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        expected = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}, {"col1": 3, "col2": "c"}]

        result = framework.transform(input_data, set())
        assert result == expected

    def test_transform_single_dict(self) -> None:
        """Test transformation from single dict to list of dicts."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        # Test single dict transformation
        input_data = {"col1": 1, "col2": "a"}
        expected = [{"col1": 1, "col2": "a"}]

        result = framework.transform(input_data, set())
        assert result == expected

    def test_transform_list_passthrough(self) -> None:
        """Test that list of dicts passes through unchanged."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        # Test list passthrough
        input_data = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]

        result = framework.transform(input_data, set())
        assert result == input_data

    def test_transform_empty_data_error(self) -> None:
        """Test that empty data raises an error."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        with pytest.raises(ValueError, match="Data cannot be empty"):
            framework.transform(None, set())

    def test_transform_invalid_data(self) -> None:
        """Test that invalid data types raise errors."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        with pytest.raises(ValueError, match="Data type .* is not supported"):
            framework.transform("invalid", set())

    def test_select_data_by_column_names(self) -> None:
        """Test column selection functionality."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        data = [{"col1": 1, "col2": "a", "col3": 10}, {"col1": 2, "col2": "b", "col3": 20}]

        feature_names = {FeatureName("col1"), FeatureName("col2")}

        result = framework.select_data_by_column_names(data, feature_names)
        expected = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]

        assert result == expected

    def test_select_data_empty_error(self) -> None:
        """Test that empty data raises an error in column selection."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        feature_names = {FeatureName("col1")}

        with pytest.raises(ValueError, match="Data cannot be empty"):
            framework.select_data_by_column_names([], feature_names)

    def test_set_column_names(self) -> None:
        """Test setting column names from data."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        framework.data = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b", "col3": 10}]

        framework.set_column_names()

        expected_columns = {"col1", "col2", "col3"}
        assert framework.column_names == expected_columns

    def test_set_column_names_empty_error(self) -> None:
        """Test that empty data raises an error when setting column names."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        framework.data = []

        with pytest.raises(ValueError, match="Data is empty or not in expected format"):
            framework.set_column_names()

    def test_merge_engine(self) -> None:
        """Test that merge engine returns correct type."""
        from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
            PythonDictMergeEngine,
        )

        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert framework.merge_engine() == PythonDictMergeEngine

    def test_filter_engine(self) -> None:
        """Test that filter engine returns correct type."""
        from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import (
            PythonDictFilterEngine,
        )

        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert framework.filter_engine() == PythonDictFilterEngine
