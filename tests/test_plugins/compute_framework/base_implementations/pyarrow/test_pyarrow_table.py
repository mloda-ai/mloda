from typing import Any, Optional, Type
import pytest
from unittest.mock import patch
import pyarrow as pa

from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_plugins.compute_framework.test_tooling.dataframe_test_base import DataFrameTestBase


class TestPyArrowTableAvailability:
    @patch("builtins.__import__")
    def test_is_available_when_pyarrow_not_installed(self, mock_import: Any) -> None:
        """Test that is_available() returns False when pyarrow import fails."""

        def side_effect(name: Any, *args: Any, **kwargs: Any) -> Any:
            if name == "pyarrow":
                raise ImportError("No module named 'pyarrow'")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect
        assert PyArrowTable.is_available() is False


class TestPyArrowTableComputeFramework:
    pyarrow_table = PyArrowTable(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
    dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
    expected_data = pa.table(dict_data)
    left_data = pa.table({"idx": [1, 3], "col1": ["a", "b"]})
    right_data = pa.table({"idx": [1, 2], "col2": ["x", "z"]})
    idx = Index(("idx",))

    def test_expected_data_framework(self) -> None:
        assert self.pyarrow_table.expected_data_framework() == pa.Table

    def test_transform_dict_to_table(self) -> None:
        assert self.pyarrow_table.transform(self.dict_data, set()) == self.expected_data

    def test_transform_arrays(self) -> None:
        chunked_array = pa.chunked_array([pa.array([1, 2]), pa.array([3])])
        pa_array = pa.array([1, 2, 3])

        for data in [chunked_array, pa_array]:
            _pytable = PyArrowTable(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
            _pytable.set_data(pa.table({"existing_column": [4, 5, 6]}))

            data = _pytable.transform(data=data, feature_names={"new_column"})
            assert data.equals(pa.table({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]}))

    def test_transform_invalid_data(self) -> None:
        with pytest.raises(ValueError):
            self.pyarrow_table.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self) -> None:
        data = self.pyarrow_table.select_data_by_column_names(self.expected_data, {FeatureName("column1")})
        assert data.schema.names == ["column1"]

    def test_set_column_names(self) -> None:
        self.pyarrow_table.data = self.expected_data
        self.pyarrow_table.set_column_names()
        assert self.pyarrow_table.column_names == {"column1", "column2"}


class TestPyArrowTableMerge(DataFrameTestBase):
    """Test PyArrowTable merge operations using the base test class."""

    @classmethod
    def framework_class(cls) -> Type[Any]:
        """Return the PyArrowTable class."""
        return PyArrowTable

    def create_dataframe(self, data: dict[str, Any]) -> Any:
        """Create a pyarrow Table from a dictionary."""
        return pa.table(data)

    def get_connection(self) -> Optional[Any]:
        """Return connection object (None for pyarrow)."""
        return None

    @pytest.mark.skip(reason="PyArrow requires matching schemas for append - base test uses different columns")
    def test_merge_append(self) -> None:
        """Skip APPEND test for PyArrow due to schema requirements."""
        pass

    @pytest.mark.skip(reason="PyArrow does not support UNION operations - see GitHub issue #30950")
    def test_merge_union(self) -> None:
        """Skip UNION test for PyArrow as it's not supported."""
        pass
