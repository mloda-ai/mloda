from typing import Any

import pytest
from unittest.mock import Mock, patch
from mloda._testing.compute_framework.dtype_suite import DataTypeValidatorSuite, DtypeExtractionSuite
from mloda._testing.compute_framework.frame_suite import AppendColumnsSuite, EmptyResultSuite
from mloda._testing.declarations import Cause
from mloda._testing.framework_adapter import FrameworkAdapter, IcebergAdapter, PyArrowAdapter, to_arrow_table
from mloda._testing.typed_data import TypedTestData
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from mloda.user import DataType
from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from tests.test_plugins.compute_framework.test_tooling.availability_test_helper import (
    assert_unavailable_when_import_blocked,
)

import logging

logger = logging.getLogger(__name__)

try:
    import pyiceberg
    import pyarrow as pa
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.catalog import Catalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import (
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        NestedField,
        StringType,
        TimestampType,
        TimestamptzType,
        ListType,
        StructType,
    )
except ImportError:
    logger.warning("PyIceberg or PyArrow is not installed. Some tests will be skipped.")
    pyiceberg = None  # type: ignore
    pa = None  # type: ignore[assignment, unused-ignore]
    IcebergTable = None  # type: ignore
    Catalog = None  # type: ignore
    Schema = None  # type: ignore
    DoubleType = None  # type: ignore
    FloatType = None  # type: ignore
    IntegerType = None  # type: ignore
    LongType = None  # type: ignore
    NestedField = None  # type: ignore
    StringType = None  # type: ignore
    TimestampType = None  # type: ignore
    TimestamptzType = None  # type: ignore
    ListType = None  # type: ignore
    StructType = None  # type: ignore


def test_real_iceberg_table_without_fixture_raises() -> None:
    from tests.test_plugins.compute_framework.base_implementations.iceberg import iceberg_test_table

    with patch.object(iceberg_test_table, "adapter_handle", []):
        with pytest.raises(RuntimeError, match="needs the iceberg_table_source fixture"):
            iceberg_test_table.real_iceberg_table(None)


class TestIcebergFrameworkAvailability:
    """Owns the is_available() contract of IcebergFramework: it needs pyiceberg AND pyarrow."""

    def test_is_available_when_pyiceberg_not_installed(self) -> None:
        """Test that is_available() returns False when pyiceberg import fails."""
        assert_unavailable_when_import_blocked(IcebergFramework, ["pyiceberg"])

    def test_is_available_when_pyarrow_not_installed(self) -> None:
        """pyiceberg alone is not enough: PyArrow is the interchange format the framework reads and
        writes, so a missing pyarrow makes it unavailable even with pyiceberg installed (issue #736)."""
        assert_unavailable_when_import_blocked(IcebergFramework, ["pyarrow"])


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergFrameworkComputeFramework:
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.iceberg_framework = IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        self.dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
        self.expected_arrow_data = pa.Table.from_pydict(self.dict_data)

    def test_is_available(self) -> None:
        """Test that is_available returns True when dependencies are installed."""
        assert IcebergFramework.is_available() is True

    def test_expected_data_framework(self) -> None:
        """Test that expected_data_framework returns IcebergTable type."""
        assert self.iceberg_framework.expected_data_framework() == IcebergTable

    def test_transform_dict_to_arrow(self) -> None:
        """Test transforming dict to PyArrow table (intermediate step for Iceberg)."""
        result = self.iceberg_framework.transform(self.dict_data, [])
        assert isinstance(result, pa.Table)
        assert result.column_names == ["column1", "column2"]
        assert result.to_pydict() == self.dict_data

    def test_transform_iceberg_table_passthrough(self, iceberg_adapter: Any) -> None:
        """Test that Iceberg tables pass through unchanged."""
        table = iceberg_adapter.from_arrow(pa.table({"a": [1]}))
        result = self.iceberg_framework.transform(table, [])
        assert result is table

    def test_transform_invalid_data(self) -> None:
        """Test that invalid data types raise ValueError."""
        with pytest.raises(ValueError, match="Data type .* is not supported"):
            self.iceberg_framework.transform(data=["invalid"], feature_names=[])

    def test_set_framework_connection_object_catalog(self, iceberg_adapter: Any) -> None:
        """Test setting a catalog as framework connection object."""
        catalog = iceberg_adapter.connection

        self.iceberg_framework.set_framework_connection_object(catalog)
        assert self.iceberg_framework.framework_connection_object is catalog

    def test_set_framework_connection_object_table(self, iceberg_adapter: Any) -> None:
        """Test setting an Iceberg table as framework connection object."""
        table = iceberg_adapter.from_arrow(pa.table({"a": [1]}))

        self.iceberg_framework.set_framework_connection_object(table)
        assert self.iceberg_framework.framework_connection_object is table

    def test_set_framework_connection_object_invalid(self) -> None:
        """Test that invalid connection objects raise ValueError."""
        with pytest.raises(ValueError, match="Expected an Iceberg catalog or table"):
            self.iceberg_framework.set_framework_connection_object("invalid")

    def test_select_data_by_column_names_pyarrow_table(self) -> None:
        """A pa.Table is column-selected like PyArrowTable does it."""
        data = pa.table({"a": [1], "b": [2], "c": [3]})
        result = self.iceberg_framework.select_data_by_column_names(data, [FeatureName("a")])
        assert result.column_names == ["a"]

    def test_select_data_by_column_names_iceberg_table(self, iceberg_adapter: Any) -> None:
        """An Iceberg Table is scanned with selected_fields and materialized in request order."""
        table = iceberg_adapter.from_arrow(pa.table({"a": [1, 2], "b": [4, 5], "c": [7, 8]}))

        with patch.object(table, "scan", wraps=table.scan) as spy:
            result = self.iceberg_framework.select_data_by_column_names(
                table,
                [FeatureName("c"), FeatureName("a")],
                column_ordering="request_order",
                request_feature_order=["c", "a"],
            )

        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert set(kwargs["selected_fields"]) == {"a", "c"}
        assert isinstance(result, pa.Table)
        assert result.column_names == ["c", "a"]
        assert result["c"].to_pylist() == [7, 8]
        assert result["a"].to_pylist() == [1, 2]

    def test_select_data_by_column_names_iceberg_table_nested_field(self, iceberg_adapter: Any) -> None:
        """A nested field path comes back as its own top-level column, named "b.c", in requested order."""
        struct_type = pa.struct([("c", pa.int64())])
        table = iceberg_adapter.from_arrow(
            pa.table(
                {
                    "id": pa.array([1], type=pa.int64()),
                    "b": pa.array([{"c": 2}], type=struct_type),
                }
            )
        )

        scanned = table.scan(selected_fields=("id", "b.c")).to_arrow()
        assert scanned.schema.field("b").type == struct_type

        with patch.object(table, "scan", wraps=table.scan) as spy:
            result = self.iceberg_framework.select_data_by_column_names(
                table,
                [FeatureName("id"), FeatureName("b.c")],
                column_ordering="request_order",
            )

        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert set(kwargs["selected_fields"]) == {"id", "b.c"}
        assert result.column_names == ["id", "b.c"]
        assert "b" not in result.column_names
        assert result["b.c"].to_pylist() == [2]
        assert result["b.c"].type == pa.int64()

    def test_set_column_names_iceberg_table(self, iceberg_adapter: Any) -> None:
        """Test setting column names from Iceberg table."""
        table = iceberg_adapter.from_arrow(pa.table({"col1": [1], "col2": [2], "col3": [3]}))

        self.iceberg_framework.data = table
        self.iceberg_framework.set_column_names()

        assert self.iceberg_framework.column_names == {"col1", "col2", "col3"}

    def test_set_column_names_no_data(self) -> None:
        """Test that None data raises an error when setting column names."""
        self.iceberg_framework.data = None
        with pytest.raises(AttributeError):
            self.iceberg_framework.set_column_names()

    def test_merge_engine_not_implemented(self) -> None:
        """Test that merge engine raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Merge functionality is not implemented"):
            self.iceberg_framework.merge_engine()

    def test_filter_engine_returns_iceberg_filter_engine(self) -> None:
        """Test that filter_engine returns IcebergFilterEngine."""
        from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_filter_engine import (
            IcebergFilterEngine,
        )

        assert self.iceberg_framework.filter_engine() == IcebergFilterEngine


@pytest.mark.skipif(
    pyiceberg is not None and pa is not None, reason="PyIceberg and PyArrow are installed. Skipping unavailable test."
)
class TestIcebergFrameworkUnavailable:
    """Test behavior when PyIceberg is not available. is_available() itself is owned by
    TestIcebergFrameworkAvailability above, which runs whether or not the libraries are installed."""

    def test_expected_data_framework_raises_when_not_installed(self) -> None:
        """Test that expected_data_framework raises ImportError when PyIceberg is not installed."""
        with patch("mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework.IcebergTable", None):
            with pytest.raises(ImportError, match="PyIceberg is not installed"):
                IcebergFramework.expected_data_framework()

    def test_set_framework_connection_object_raises_when_not_installed(self) -> None:
        """Test that set_framework_connection_object raises ImportError when PyIceberg is not installed."""
        framework = IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        with patch("mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework.Catalog", None):
            with pytest.raises(ImportError, match="PyIceberg is not installed"):
                framework.set_framework_connection_object(Mock())


def _arrow_frame(suite: Any, name: str, data: TypedTestData) -> Any:
    """The post-transform shape: a PyArrow table, as IcebergFramework.transform emits for dict input."""
    return to_arrow_table(data, suite.arrow_types(name))


_NO_DUPLICATES = "Iceberg schemas cannot carry duplicate column names"


class TestIcebergDtypeExtractionSuite(DtypeExtractionSuite):
    adapter_class = IcebergAdapter
    switched_off = {
        "test_extract_duplicate_column_dtype_returns_first_occurrence": Cause.BACKEND,
        "test_extract_duplicate_column_data_type_returns_first_occurrence": Cause.BACKEND,
        "test_extract_duplicate_column_dtype_reversed_order_returns_first_occurrence": Cause.BACKEND,
        "test_extract_duplicate_column_data_type_reversed_order_returns_first_occurrence": Cause.BACKEND,
        "test_extract_column_dtype_consistent_with_output_schema": Cause.BACKEND,
    }

    @pytest.mark.skip(reason=_NO_DUPLICATES)
    def test_extract_duplicate_column_dtype_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason=_NO_DUPLICATES)
    def test_extract_duplicate_column_data_type_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason=_NO_DUPLICATES)
    def test_extract_duplicate_column_dtype_reversed_order_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason=_NO_DUPLICATES)
    def test_extract_duplicate_column_data_type_reversed_order_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason=_NO_DUPLICATES)
    def test_extract_column_dtype_consistent_with_output_schema(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...


class TestIcebergDtypeExtractionPyArrowSuite(DtypeExtractionSuite):
    """Pin dtype extraction for the post-transform PyArrow shape, not just a native IcebergTable."""

    adapter_class = IcebergAdapter

    def frame(self, adapter: FrameworkAdapter, name: str, data: TypedTestData) -> Any:
        return _arrow_frame(self, name, data)

    def duplicate_column_frame(self, adapter: FrameworkAdapter, first_type: DataType, second_type: DataType) -> Any:
        return super().duplicate_column_frame(PyArrowAdapter(), first_type, second_type)


_NO_UNSIGNED = (Cause.BACKEND, "Iceberg has no unsigned integer types")
_NO_SMALL_INT = (Cause.BACKEND, "Iceberg has no 8-bit or 16-bit integer type")
_NO_NS = (Cause.BACKEND, "Iceberg does not support nanosecond timestamp precision")
_ONE_TIMESTAMP = "Iceberg has only one TimestampType (microseconds per spec); millisecond cannot be expressed"


class TestIcebergDataTypeValidatorSuite(DataTypeValidatorSuite):
    adapter_class = IcebergAdapter
    unsupported_cases = {
        "int8_col": _NO_SMALL_INT,
        "uint8_col": _NO_UNSIGNED,
        "uint64_col": _NO_UNSIGNED,
        "int16_col": _NO_SMALL_INT,
        "uint16_col": _NO_UNSIGNED,
        "uint32_col": _NO_UNSIGNED,
        "timestamp_ns_col": _NO_NS,
        "timestamp_ns_tz_col": _NO_NS,
    }
    switched_off = {
        "test_timestamp_ms_column_strict_ms_passes": Cause.BACKEND,
        "test_timestamp_us_column_strict_ms_raises": Cause.BACKEND,
    }

    @pytest.mark.skip(reason=_ONE_TIMESTAMP)
    def test_timestamp_ms_column_strict_ms_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None: ...

    @pytest.mark.skip(reason=_ONE_TIMESTAMP)
    def test_timestamp_us_column_strict_ms_raises(self, adapter: FrameworkAdapter, framework_instance: Any) -> None: ...


class TestIcebergDataTypeValidatorPyArrowSuite(DataTypeValidatorSuite):
    """Pin DataTypeValidator enforcement for the post-transform PyArrow shape."""

    adapter_class = IcebergAdapter

    def frame(self, adapter: FrameworkAdapter, name: str, data: TypedTestData) -> Any:
        return _arrow_frame(self, name, data)


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
def test_extract_column_dtype_pyarrow_table_returns_arrow_type_string() -> None:
    framework = IcebergFramework()
    assert framework._extract_column_dtype(pa.table({"a": [1]}), "a") == "int64"


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
def test_extract_column_dtype_pyarrow_table_missing_column_returns_none() -> None:
    framework = IcebergFramework()
    assert framework._extract_column_dtype(pa.table({"a": [1]}), "unknown") is None


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
def test_extract_column_data_type_pyarrow_table_returns_int64() -> None:
    framework = IcebergFramework()
    assert framework._extract_column_data_type(pa.table({"a": [1]}), "a") is DataType.INT64


class TestIcebergEmptyResultSuite(EmptyResultSuite):
    """Both ``extract_column_names`` branches: zero-row and appended data are post-transform Arrow tables, non_empty is native."""

    adapter_class = IcebergAdapter

    def frame(self, adapter: FrameworkAdapter, name: str, data: TypedTestData) -> Any:
        if name == "non_empty":
            return super().frame(adapter, name, data)
        return _arrow_frame(self, name, data)


class TestIcebergAppendColumnsSuite(AppendColumnsSuite):
    """``_append_columns`` only serves the post-transform Arrow shape, so frames are Arrow tables."""

    adapter_class = IcebergAdapter
    switched_off = {
        "test_single_column_on_unpinned_joined_frame": Cause.NOT_IMPLEMENTED,
        "test_single_column_on_pinned_frame_appends_aligned": Cause.NOT_IMPLEMENTED,
    }

    def frame(self, adapter: FrameworkAdapter, name: str, data: TypedTestData) -> Any:
        return _arrow_frame(self, name, data)

    @pytest.mark.skip(reason="Iceberg has no single-column type")
    def test_single_column_on_unpinned_joined_frame(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason="Iceberg has no single-column type")
    def test_single_column_on_pinned_frame_appends_aligned(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergEmptyResult:
    def test_append_columns_on_iceberg_table_returns_none(self, iceberg_adapter: Any) -> None:
        framework = IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        table = iceberg_adapter.from_arrow(pa.table({"a": [1, 2]}))
        assert framework._append_columns(table, {"appended_col": [1]}) is None


from tests.test_plugins.compute_framework.base_implementations.tfs_connection_test_mixin import TfsConnectionInitMixin  # noqa: E402


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergTfsConnectionInit(TfsConnectionInitMixin):
    _SKIP = "Iceberg data does not carry its own connection; no per-data connection check"

    def test_adopt_connection_rejects_data_on_another_connection_than_the_bound_one(
        self, framework_class: Any, valid_connection: Any, second_valid_connection: Any
    ) -> None:
        pytest.skip(self._SKIP)

    def test_adopt_connection_accepts_the_bound_connection_and_adopts_when_unbound(
        self, framework_class: Any, valid_connection: Any
    ) -> None:
        pytest.skip(self._SKIP)

    def test_missing_connection_error_names_the_dac_and_connection_handle(self, framework_class: Any) -> None:
        pytest.skip(self._SKIP)

    @pytest.fixture
    def framework_class(self) -> Any:
        return IcebergFramework

    @pytest.fixture
    def valid_connection(self) -> Any:
        mock_catalog = Mock(spec=Catalog)
        mock_catalog.load_table = Mock()
        return mock_catalog

    @pytest.fixture
    def second_valid_connection(self) -> Any:
        mock_catalog = Mock(spec=Catalog)
        mock_catalog.load_table = Mock()
        return mock_catalog


@pytest.mark.skipif(pyiceberg is None, reason="PyIceberg is not installed. Skipping this test.")
class TestIcebergIsMissingColumnError:
    def test_schema_projection_of_an_unknown_column_is_a_missing_column(self) -> None:
        schema = Schema(NestedField(field_id=1, name="a", field_type=LongType(), required=False))
        with pytest.raises(ValueError) as excinfo:
            schema.select("x")
        assert IcebergFramework.is_missing_column_error(excinfo.value) is True

    def test_plain_could_not_find_column_without_key_error_cause_is_not_a_missing_column(self) -> None:
        assert IcebergFramework.is_missing_column_error(ValueError("Could not find column: x")) is False

    def test_mloda_explanatory_error_chained_from_key_error_is_not_a_missing_column(self) -> None:
        try:
            raise KeyError("x")
        except KeyError as key_error:
            explanatory = ValueError("Feature 'f' read a column it did not declare in input_features().")
            explanatory.__cause__ = key_error
        assert IcebergFramework.is_missing_column_error(explanatory) is False
