from typing import Any, Optional, Type
import pytest
import pyarrow as pa

from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_plugins.compute_framework.test_tooling.dataframe_test_base import DataFrameTestBase
from tests.test_plugins.compute_framework.test_tooling.availability_test_helper import (
    assert_unavailable_when_import_blocked,
)


class TestPyArrowTableAvailability:
    def test_is_available_when_pyarrow_not_installed(self) -> None:
        """Test that is_available() returns False when pyarrow import fails."""
        assert_unavailable_when_import_blocked(PyArrowTable, ["pyarrow"])


class TestPyArrowTableComputeFramework:
    @pytest.fixture
    def pyarrow_table(self) -> PyArrowTable:
        """Create a fresh PyArrowTable instance for each test."""
        return PyArrowTable(mode=ParallelizationModes.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def expected_data(self, dict_data: dict[str, list[int]]) -> Any:
        """Create fresh expected PyArrow table for each test."""
        return pa.table(dict_data)

    def test_expected_data_framework(self, pyarrow_table: PyArrowTable) -> None:
        assert pyarrow_table.expected_data_framework() == pa.Table

    def test_transform_dict_to_table(
        self, pyarrow_table: PyArrowTable, dict_data: dict[str, list[int]], expected_data: Any
    ) -> None:
        assert pyarrow_table.transform(dict_data, set()) == expected_data

    def test_transform_arrays(self) -> None:
        chunked_array = pa.chunked_array([pa.array([1, 2]), pa.array([3])])
        pa_array = pa.array([1, 2, 3])

        for data in [chunked_array, pa_array]:
            _pytable = PyArrowTable(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
            _pytable.set_data(pa.table({"existing_column": [4, 5, 6]}))

            data = _pytable.transform(data=data, feature_names={"new_column"})
            assert data.equals(pa.table({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]}))

    def test_transform_invalid_data(self, pyarrow_table: PyArrowTable) -> None:
        with pytest.raises(ValueError):
            pyarrow_table.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self, pyarrow_table: PyArrowTable, expected_data: Any) -> None:
        data = pyarrow_table.select_data_by_column_names(expected_data, {FeatureName("column1")})
        assert data.schema.names == ["column1"]

    def test_set_column_names(self, pyarrow_table: PyArrowTable, expected_data: Any) -> None:
        pyarrow_table.data = expected_data
        pyarrow_table.set_column_names()
        assert pyarrow_table.column_names == {"column1", "column2"}


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
