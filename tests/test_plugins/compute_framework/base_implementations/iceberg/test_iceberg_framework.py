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
)
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
    )
except ImportError:
    logger.warning("PyIceberg or PyArrow is not installed. Some tests will be skipped.")
    pyiceberg = None  # type: ignore
    pa = None  # type: ignore[assignment]
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
        result = self.iceberg_framework.transform(self.dict_data, set())
        assert isinstance(result, pa.Table)
        assert result.column_names == ["column1", "column2"]
        assert result.to_pydict() == self.dict_data

    def test_transform_iceberg_table_passthrough(self) -> None:
        """Test that Iceberg tables pass through unchanged."""
        mock_iceberg_table = Mock(spec=IcebergTable)
        result = self.iceberg_framework.transform(mock_iceberg_table, set())
        assert result is mock_iceberg_table

    def test_transform_invalid_data(self) -> None:
        """Test that invalid data types raise ValueError."""
        with pytest.raises(ValueError, match="Data type .* is not supported"):
            self.iceberg_framework.transform(data=["invalid"], feature_names=set())

    def test_set_framework_connection_object_catalog(self) -> None:
        """Test setting a catalog as framework connection object."""
        mock_catalog = Mock(spec=Catalog)
        mock_catalog.load_table = Mock()

        self.iceberg_framework.set_framework_connection_object(mock_catalog)
        assert self.iceberg_framework.framework_connection_object is mock_catalog

    def test_set_framework_connection_object_table(self) -> None:
        """Test setting an Iceberg table as framework connection object."""
        mock_table = Mock(spec=IcebergTable)

        self.iceberg_framework.set_framework_connection_object(mock_table)
        assert self.iceberg_framework.framework_connection_object is mock_table

    def test_set_framework_connection_object_invalid(self) -> None:
        """Test that invalid connection objects raise ValueError."""
        with pytest.raises(ValueError, match="Expected an Iceberg catalog or table"):
            self.iceberg_framework.set_framework_connection_object("invalid")

    def test_select_data_by_column_names_non_iceberg(self) -> None:
        """Test that non-Iceberg data passes through unchanged."""
        data = "not_iceberg_table"
        feature_names = {FeatureName("column1")}
        result = self.iceberg_framework.select_data_by_column_names(data, feature_names)
        assert result is data

    def test_set_column_names_iceberg_table(self) -> None:
        """Test setting column names from Iceberg table."""
        mock_table = Mock(spec=IcebergTable)
        mock_schema = Mock()
        mock_schema.column_names = ["col1", "col2", "col3"]
        mock_table.schema.return_value = mock_schema

        self.iceberg_framework.data = mock_table
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
    """Test behavior when PyIceberg is not available."""

    def test_is_available_false_when_not_installed(self) -> None:
        """Test that is_available returns False when dependencies are not installed."""
        assert_unavailable_when_import_blocked(IcebergFramework, ["pyiceberg"])

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

    Iceberg tables require catalog context to construct, so the fixture wraps a real
    ``pyiceberg.schema.Schema`` (carrying real ``IntegerType``/``LongType``/... field
    types) inside a ``Mock`` IcebergTable. The mock's ``schema()`` returns a wrapper
    that adapts ``find_field`` to return ``None`` for missing columns (real pyiceberg
    raises ``ValueError`` here; the framework code assumes ``None``).

    Iceberg has only one TimestampType (microsecond precision per spec), so the
    millisecond-precision tests are skipped on this subclass.
    """

    @staticmethod
    def _wrap_schema(schema: Any) -> Any:
        """Wrap a real pyiceberg Schema so that find_field returns None on missing."""
        wrapper = Mock()
        wrapper.column_names = list(schema.column_names)

        def _find_field_safe(name: str) -> Any:
            for field in schema.fields:
                if field.name == name:
                    return field
            return None

        wrapper.find_field = _find_field_safe
        mock_table = Mock(spec=IcebergTable)
        mock_table.schema.return_value = wrapper
        return mock_table

    @pytest.fixture
    def framework_instance(self) -> Any:
        return IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def validator_sample_data(self) -> Any:
        return self._build_iceberg(self.VALIDATOR_COLUMNS)

    @pytest.fixture
    def precision_sample_data(self) -> Any:
        return self._build_iceberg(self.PRECISION_COLUMNS)

    def _build_iceberg(self, columns: tuple[ColumnSpec, ...]) -> Any:
        # Iceberg needs explicit NestedField/IcebergType per column; the schema carries no
        # values. _ICEBERG_TYPE_MAP omits TIMESTAMP_MILLIS, so unsupported columns are
        # filtered out (the MILLIS test methods skip explicitly).
        fields = [
            NestedField(i + 1, c.name, _ICEBERG_TYPE_MAP[c.data_type])
            for i, c in enumerate(columns)
            if c.data_type in _ICEBERG_TYPE_MAP
        ]
        return self._wrap_schema(Schema(*fields))

    def test_timestamp_ms_column_strict_ms_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Iceberg has only one TimestampType (microseconds per spec); millisecond cannot be expressed")

    def test_timestamp_us_column_strict_ms_raises(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Iceberg has only one TimestampType (microseconds per spec); millisecond cannot be expressed")


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergDtypeExtraction(DtypeExtractionTestMixin):
    """Test IcebergFramework._extract_column_dtype using shared mixin.

    Iceberg tables need catalog context to construct, so the fixture wraps a real
    ``pyiceberg.schema.Schema`` (carrying real ``LongType``/``StringType``/``DoubleType``
    field types) inside a ``Mock`` IcebergTable, reusing TestIcebergDataTypeValidator's
    ``_wrap_schema`` helper.
    """

    @pytest.fixture
    def framework_instance(self) -> Any:
        return IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def dtype_sample_data(self) -> Any:
        schema = Schema(
            NestedField(1, "int_col", LongType()),
            NestedField(2, "str_col", StringType()),
            NestedField(3, "float_col", DoubleType()),
        )
        return TestIcebergDataTypeValidator._wrap_schema(schema)


from tests.test_plugins.compute_framework.base_implementations.tfs_connection_test_mixin import TfsConnectionInitMixin  # noqa: E402


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergTfsConnectionInit(TfsConnectionInitMixin):
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
