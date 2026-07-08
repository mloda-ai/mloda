from typing import Any, Optional
import pytest
import pyarrow as pa

from mloda.provider import EmptyResultError
from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_plugins.compute_framework.test_tooling.dataframe_test_base import DataFrameTestBase
from tests.test_plugins.compute_framework.test_tooling.availability_test_helper import (
    assert_unavailable_when_import_blocked,
)
from tests.test_plugins.compute_framework.base_implementations.datatype_validator_test_mixin import (
    DataTypeValidatorFrameworkTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.dtype_extraction_test_mixin import (
    DtypeExtractionTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.empty_result_test_mixin import (
    EmptyResultFrameworkTestMixin,
)


class TestPyArrowTableAvailability:
    def test_is_available_when_pyarrow_not_installed(self) -> None:
        """Test that is_available() returns False when pyarrow import fails."""
        assert_unavailable_when_import_blocked(PyArrowTable, ["pyarrow"])


class TestPyArrowTableComputeFramework:
    @pytest.fixture
    def pyarrow_table(self) -> PyArrowTable:
        """Create a fresh PyArrowTable instance for each test."""
        return PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def expected_data(self, dict_data: dict[str, list[int]]) -> Any:
        """Create fresh expected PyArrow table for each test."""
        return pa.table(dict_data)

    def test_expected_data_framework(self, pyarrow_table: PyArrowTable) -> None:
        assert pyarrow_table.expected_data_framework() == pa.Table

    def test_transform_dict_to_table(
        self, pyarrow_table: PyArrowTable, dict_data: dict[str, list[int]], expected_data: Any
    ) -> None:
        assert pyarrow_table.transform(dict_data, []) == expected_data

    def test_transform_arrays(self) -> None:
        chunked_array = pa.chunked_array([pa.array([1, 2]), pa.array([3])])
        pa_array = pa.array([1, 2, 3])

        for data in [chunked_array, pa_array]:
            _pytable = PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
            _pytable.set_data(pa.table({"existing_column": [4, 5, 6]}))

            data = _pytable.transform(data=data, feature_names=["new_column"])
            assert data.equals(pa.table({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]}))

    def test_transform_invalid_data(self, pyarrow_table: PyArrowTable) -> None:
        with pytest.raises(ValueError):
            pyarrow_table.transform(data=["a"], feature_names=[])

    def test_select_data_by_column_names(self, pyarrow_table: PyArrowTable, expected_data: Any) -> None:
        data = pyarrow_table.select_data_by_column_names(expected_data, [FeatureName("column1")])
        assert data.schema.names == ["column1"]

    def test_set_column_names(self, pyarrow_table: PyArrowTable, expected_data: Any) -> None:
        pyarrow_table.data = expected_data
        pyarrow_table.set_column_names()
        assert pyarrow_table.column_names == {"column1", "column2"}


class TestPyArrowTableMerge(DataFrameTestBase):
    """Test PyArrowTable merge operations using the base test class."""

    @classmethod
    def framework_class(cls) -> type[Any]:
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


class TestPyArrowDtypeExtraction(DtypeExtractionTestMixin):
    """Test PyArrowTable._extract_column_dtype using shared mixin."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def dtype_sample_data(self) -> Any:
        return pa.table({"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.0, 2.0, 3.0]})


class TestPyArrowDataTypeValidator(DataTypeValidatorFrameworkTestMixin):
    """Test DataTypeValidator enforcement on PyArrowTable using shared mixin."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    def from_arrow(self, table: pa.Table) -> pa.Table:
        return table


class TestPyArrowEmptyResult(EmptyResultFrameworkTestMixin):
    """Test PyArrowTable schema detection via shared mixin (zero-row table keeps columns)."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def empty_data(self) -> Any:
        return pa.table({"a": pa.array([], pa.int64())})

    @pytest.fixture
    def non_empty_data(self) -> Any:
        return pa.table({"a": [1]})


class _FakeGuardFeatureGroup:
    """Minimal FeatureGroup stand-in for the empty-result guard.

    ``run_validate_output_features`` uses ``get_class_name()`` for the error message and
    finally calls ``validate_output_features`` (no-op). The guard now keys purely on schema
    presence (zero columns), so there is no opt-in flag to declare.
    """

    def get_class_name(self) -> str:
        return "FakeGuardFeatureGroup"

    def validate_output_features(self, data: Any, features: Any) -> None:
        return None


class _FakeGuardFeatures:
    """Minimal FeatureSet stand-in for the empty-result guard.

    The guard reads ``get_initial_requested_features()`` (non-empty so the guard
    applies to a final feature). Past the guard, ``DataTypeValidator.validate``
    iterates ``features.features``; an empty list makes validation a no-op.
    """

    def __init__(self) -> None:
        self.features: list[Any] = []

    def get_initial_requested_features(self) -> set[str]:
        return {"a"}


class TestPyArrowEmptyResultGuard:
    """Direct unit tests for the EmptyResultError guard on a schema-bearing framework.

    The guard firing end-to-end is only exercised via python_dict; this pins the guard
    contract directly on PyArrow: zero COLUMNS raises, zero ROWS with a schema does not.
    """

    def test_zero_columns_raises_empty_result_error(self) -> None:
        """A schema-less result (zero-column table) trips the guard for a final feature."""
        framework = PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        framework.data = pa.table({})
        # run_calculation always sets column names before validating; the direct call mirrors that sequence.
        framework.set_column_names()

        with pytest.raises(EmptyResultError):
            framework.run_validate_output_features(_FakeGuardFeatureGroup(), _FakeGuardFeatures())

    def test_zero_rows_with_schema_does_not_raise(self) -> None:
        """A zero-row but column-bearing table is a valid result; the guard must not fire.

        The call proceeds through DataTypeValidator.validate (no-op: no features with
        declared data types) and validate_output_features (no-op fake).
        """
        framework = PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        framework.data = pa.table({"a": pa.array([], pa.int64())})
        # run_calculation always sets column names before validating; the direct call mirrors that sequence.
        framework.set_column_names()

        # Must not raise.
        framework.run_validate_output_features(_FakeGuardFeatureGroup(), _FakeGuardFeatures())
