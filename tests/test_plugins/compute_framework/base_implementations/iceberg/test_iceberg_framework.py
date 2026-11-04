from decimal import Decimal
from typing import Any

import pytest
from unittest.mock import Mock, patch
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from mloda.user import DataType
from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from tests.test_plugins.compute_framework.base_implementations.datatype_validator_test_mixin import (
    ColumnSpec,
    DataTypeValidatorFrameworkTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.dtype_extraction_test_mixin import (
    DtypeExtractionTestMixin,
    DuplicateColumnDtypeExtractionTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.empty_result_test_mixin import (
    AppendColumnsFrameworkTestMixin,
    EmptyResultFrameworkTestMixin,
)
from tests.test_plugins.compute_framework.test_tooling.availability_test_helper import (
    assert_unavailable_when_import_blocked,
)

import logging
import uuid

logger = logging.getLogger(__name__)

try:
    import pyiceberg
    import pyarrow as pa
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.catalog import Catalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import (
        DecimalType,
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


def _table_from_schema(adapter: Any, schema: Any) -> Any:
    """Create a real, empty Iceberg table (no append) from a hand-built pyiceberg Schema."""
    catalog = adapter.connection
    return catalog.create_table(f"{adapter.namespace}.t_{uuid.uuid4().hex}", schema=schema)


_ICEBERG_TYPE_MAP: dict[DataType, Any] = (
    {
        DataType.INT32: IntegerType(),
        DataType.INT64: LongType(),
        DataType.FLOAT: FloatType(),
        DataType.DOUBLE: DoubleType(),
        DataType.STRING: StringType(),
        DataType.TIMESTAMP_MICROS: TimestampType(),
    }
    if pyiceberg is not None
    else {}
)


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


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergDataTypeValidator(DataTypeValidatorFrameworkTestMixin):
    """Test DataTypeValidator enforcement on IcebergFramework using shared mixin.

    Each fixture creates a real, empty Iceberg table in the adapter's catalog from a
    hand-built ``pyiceberg.schema.Schema`` (real ``IntegerType``/``LongType``/... field types).

    Iceberg has only one TimestampType (microsecond precision per spec), so the
    millisecond-precision tests are skipped on this subclass.
    """

    @pytest.fixture
    def framework_instance(self) -> Any:
        return IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    EXTENDED_SKIPS = {
        "int8_col": "Iceberg has no 8-bit integer type",
        "uint8_col": "Iceberg has no unsigned integer types",
        "uint64_col": "Iceberg has no unsigned integer types",
        "int16_col": "Iceberg has no 16-bit integer type",
        "uint16_col": "Iceberg has no unsigned integer types",
        "uint32_col": "Iceberg has no unsigned integer types",
    }

    @pytest.fixture
    def extended_sample_data(self, spec: ColumnSpec, iceberg_adapter: Any) -> Any:
        self._skip_if_unsupported(spec)
        return self._build_iceberg(iceberg_adapter, (spec,))

    @pytest.fixture
    def validator_sample_data(self, iceberg_adapter: Any) -> Any:
        return self._build_iceberg(iceberg_adapter, self.VALIDATOR_COLUMNS)

    @pytest.fixture
    def precision_sample_data(self, iceberg_adapter: Any) -> Any:
        return self._build_iceberg(iceberg_adapter, self.PRECISION_COLUMNS)

    @staticmethod
    def _iceberg_type(spec: ColumnSpec) -> Any:
        # Extended columns derive their Iceberg type from the explicit Arrow type (ListType needs an element type).
        if spec.data_type == DataType.TIMESTAMP_MICROS_TZ:
            return TimestamptzType()
        if spec.data_type == DataType.LIST:
            element = spec.arrow_type.value_type  # type: ignore[union-attr]
            if element == pa.string():
                return ListType(100, StringType(), element_required=False)
            if element == pa.int64():
                return ListType(100, LongType(), element_required=False)
            struct = StructType(NestedField(101, "x", LongType(), required=False))
            return ListType(100, struct, element_required=False)
        return _ICEBERG_TYPE_MAP[spec.data_type]

    def _build_iceberg(self, adapter: Any, columns: tuple[ColumnSpec, ...]) -> Any:
        # Iceberg needs explicit NestedField/IcebergType per column; the schema carries no
        # values. _ICEBERG_TYPE_MAP omits TIMESTAMP_MILLIS, so unsupported columns are
        # filtered out (the MILLIS test methods skip explicitly).
        fields = [
            NestedField(i + 1, c.name, self._iceberg_type(c))
            for i, c in enumerate(columns)
            if c.arrow_type is not None or c.data_type in _ICEBERG_TYPE_MAP
        ]
        return _table_from_schema(adapter, Schema(*fields))

    def test_timestamp_ms_column_strict_ms_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Iceberg has only one TimestampType (microseconds per spec); millisecond cannot be expressed")

    def test_timestamp_us_column_strict_ms_raises(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Iceberg has only one TimestampType (microseconds per spec); millisecond cannot be expressed")


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergDtypeExtraction(DtypeExtractionTestMixin):
    """Test IcebergFramework._extract_column_dtype using shared mixin.

    Each fixture builds a real
    ``pyiceberg.schema.Schema`` (carrying real ``LongType``/``StringType``/``DoubleType``
    field types) in a real, empty Iceberg table created through the adapter.
    """

    @pytest.fixture
    def framework_instance(self) -> Any:
        return IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def dtype_sample_data(self, iceberg_adapter: Any) -> Any:
        schema = Schema(
            NestedField(1, "int_col", LongType()),
            NestedField(2, "str_col", StringType()),
            NestedField(3, "float_col", DoubleType()),
        )
        return _table_from_schema(iceberg_adapter, schema)

    @pytest.fixture
    def decimal_sample_data(self, iceberg_adapter: Any) -> Any:
        return _table_from_schema(iceberg_adapter, Schema(NestedField(1, "d", DecimalType(10, 2))))


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergDtypeExtractionPyArrow(DtypeExtractionTestMixin, DuplicateColumnDtypeExtractionTestMixin):
    """Pin _extract_column_dtype for the post-transform PyArrow shape, not just a native IcebergTable."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def dtype_sample_data(self) -> Any:
        return pa.table({"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.0, 2.0, 3.0]})

    @pytest.fixture
    def decimal_sample_data(self) -> Any:
        values = [Decimal("12.34"), Decimal("5.50"), Decimal("99.99"), None]
        return pa.table({"d": pa.array(values, type=pa.decimal128(10, 2))})

    @pytest.fixture
    def dtype_duplicate_column_data(self) -> Any:
        table = pa.table({"dup_col": [1, 2, 3]})
        return table.append_column("dup_col", pa.array(["x", "y", "z"]))

    @pytest.fixture
    def dtype_duplicate_column_data_reversed(self) -> Any:
        table = pa.table({"dup_col": ["x", "y", "z"]})
        return table.append_column("dup_col", pa.array([1, 2, 3]))


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergDataTypeValidatorPyArrow(DataTypeValidatorFrameworkTestMixin):
    """Pin DataTypeValidator enforcement for the post-transform PyArrow shape, where from_arrow is the identity."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    def from_arrow(self, table: Any) -> Any:
        return table


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


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergEmptyResult(AppendColumnsFrameworkTestMixin, EmptyResultFrameworkTestMixin):
    """Test IcebergFramework schema detection via shared mixin, covering both branches.

    ``IcebergFramework.extract_column_names`` has two branches and the fixtures exercise
    one each:

    - ``empty_data`` is a real zero-row PyArrow table. ``IcebergFramework.transform`` emits
      a pa.Table for dict input, so post-transform working data takes the pa.Table branch
      (``set(data.schema.names)``); this pins that a zero-row pa.Table still exposes its
      schema (state C).
    - ``non_empty_data`` is a real native IcebergTable built through the adapter, pinning
      the native-table branch's access pattern.
    """

    @pytest.fixture
    def framework_instance(self) -> Any:
        return IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def empty_data(self) -> Any:
        return pa.table({"a": pa.array([], pa.int64())})

    @pytest.fixture
    def non_empty_data(self, iceberg_adapter: Any) -> Any:
        return iceberg_adapter.from_arrow(pa.table({"a": [1, 2]}))

    @pytest.fixture
    def appendable_data(self) -> Any:
        return pa.table({"a": [1, 2]})

    def test_append_columns_on_iceberg_table_returns_none(self, framework_instance: Any, non_empty_data: Any) -> None:
        assert framework_instance._append_columns(non_empty_data, {"appended_col": [1]}) is None


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
