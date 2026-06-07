"""
Spark Framework Tests

This module contains comprehensive tests for the Spark compute framework implementation.

Requirements:
- PySpark must be installed (pip install pyspark)
- Java 8+ must be installed and JAVA_HOME environment variable must be set

Environment Setup:
- JAVA_HOME: Must point to a valid Java installation
- SKIP_SPARK_INSTALLATION_TEST: Set to "true" to skip installation validation tests

Test Structure:
- TestSparkFrameworkAvailability: Tests framework availability detection
- TestSparkInstallation: Validates PySpark and Java setup
- TestSparkFrameworkComputeFramework: Core framework functionality tests

The tests use a shared SparkSession fixture to avoid Java gateway conflicts and
ensure proper resource management across all test methods.
"""

import os
from typing import Any
from mloda.user import DataType
from mloda.user import JoinType
import pytest
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework
from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from mloda.user import Index
from tests.test_plugins.compute_framework.test_tooling.availability_test_helper import (
    assert_unavailable_when_import_blocked,
)
from tests.test_plugins.compute_framework.base_implementations.datatype_validator_test_mixin import (
    ColumnSpec,
    DataTypeValidatorFrameworkTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.dtype_extraction_test_mixin import (
    DtypeExtractionTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.empty_result_test_mixin import (
    EmptyResultFrameworkTestMixin,
)

# Import shared fixtures and availability flags from conftest.py
from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link

import logging

logger = logging.getLogger(__name__)

# Import PySpark types for type checking (only if available)
if PYSPARK_AVAILABLE:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        IntegerType,
        LongType,
        FloatType,
        DoubleType,
        TimestampType,
    )
    import pyspark
else:
    SparkSession = None
    StructType = None
    StructField = None
    StringType = None
    IntegerType = None
    LongType = None
    FloatType = None
    DoubleType = None
    TimestampType = None
    pyspark = None


_SPARK_TYPE_MAP: dict[DataType, Any] = (
    {
        DataType.INT32: IntegerType(),
        DataType.INT64: LongType(),
        DataType.FLOAT: FloatType(),
        DataType.DOUBLE: DoubleType(),
        DataType.STRING: StringType(),
        DataType.TIMESTAMP_MICROS: TimestampType(),
    }
    if PYSPARK_AVAILABLE
    else {}
)


class TestSparkFrameworkAvailability:
    def test_is_available_when_pyspark_not_installed(self) -> None:
        """Test that is_available() returns False when pyspark import fails."""
        assert_unavailable_when_import_blocked(SparkFramework, ["pyspark.sql"])


class TestSparkInstallation:
    @pytest.mark.skipif(
        os.getenv("SKIP_SPARK_INSTALLATION_TEST", "false").lower() == "true",
        reason="Spark installation test is disabled by environment variable",
    )
    def test_spark_is_installed(self, spark_session: Any) -> None:
        """Test that PySpark is properly installed and can be imported."""
        try:
            import pyarrow as pa  # noqa: F401

            # Test basic functionality using the shared spark_session fixture
            data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
            df = spark_session.createDataFrame(data)
            result = df.collect()
            assert len(result) == 2
        except ImportError:
            pytest.fail("PySpark is not installed but is required for this test environment")


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkFrameworkComputeFramework:
    def test_expected_data_framework(self, spark_session: Any) -> None:
        from pyspark.sql import DataFrame

        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert spark_framework.expected_data_framework() == DataFrame

    def test_transform_dict_to_dataframe(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
        result = spark_framework.transform(dict_data, set())
        result_data = result.collect()

        expected_data = spark_session.createDataFrame(
            [{"column1": 1, "column2": 4}, {"column1": 2, "column2": 5}, {"column1": 3, "column2": 6}]
        )
        expected_data_collected = expected_data.collect()

        # Compare the data (order might be different)
        assert len(result_data) == len(expected_data_collected)
        assert set(result.columns) == set(expected_data.columns)

    def test_transform_invalid_data(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        with pytest.raises(ValueError):
            spark_framework.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        expected_data = spark_session.createDataFrame(
            [{"column1": 1, "column2": 4}, {"column1": 2, "column2": 5}, {"column1": 3, "column2": 6}]
        )
        data = spark_framework.select_data_by_column_names(expected_data, {FeatureName("column1")})
        assert data.columns == ["column1"]

    def test_set_column_names(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        expected_data = spark_session.createDataFrame(
            [{"column1": 1, "column2": 4}, {"column1": 2, "column2": 5}, {"column1": 3, "column2": 6}]
        )
        spark_framework.data = expected_data
        spark_framework.set_column_names()
        assert spark_framework.column_names == {"column1", "column2"}

    def test_merge_inner(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        spark_framework.data = left_data
        merge_engine_class = spark_framework.merge_engine()
        framework_connection = spark_framework.get_framework_connection_object()
        merge_engine = merge_engine_class(framework_connection)
        result = merge_engine.merge(left_data, right_data, make_merge_link(JoinType.INNER, idx, idx))

        # Check that we got a result and it has the expected structure
        assert result is not None
        result_count = result.count()
        assert result_count == 1  # Should have 1 matching row

    def test_merge_left(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        spark_framework.data = left_data
        framework_connection = spark_framework.get_framework_connection_object()
        merge_engine_class = spark_framework.merge_engine()
        merge_engine = merge_engine_class(framework_connection)
        result = merge_engine.merge(left_data, right_data, make_merge_link(JoinType.LEFT, idx, idx))

        # Check that we got a result with all left rows
        assert result is not None
        result_count = result.count()
        assert result_count == 2  # Should have 2 rows (all from left)

    def test_merge_append(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        spark_framework.data = left_data
        framework_connection = spark_framework.get_framework_connection_object()
        merge_engine_class = spark_framework.merge_engine()
        merge_engine = merge_engine_class(framework_connection)
        result = merge_engine.merge(left_data, right_data, make_merge_link(JoinType.APPEND, idx, idx))

        # Check that we got a result with combined rows
        assert result is not None
        result_count = result.count()
        assert result_count == 4  # Should have 2 + 2 rows

    def test_merge_union(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        spark_framework.data = left_data
        framework_connection = spark_framework.get_framework_connection_object()
        merge_engine_class = spark_framework.merge_engine()
        merge_engine = merge_engine_class(framework_connection)
        result = merge_engine.merge(left_data, right_data, make_merge_link(JoinType.UNION, idx, idx))

        # Check that we got a result (union removes duplicates)
        assert result is not None
        result_count = result.count()
        # The exact count depends on duplicate handling, but should be <= 4
        assert result_count <= 4

    def test_framework_connection_object(self, spark_session: Any) -> None:
        """Test that framework connection object is properly set and retrieved."""
        framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        framework.set_framework_connection_object(spark_session)

        connection = framework.get_framework_connection_object()
        assert connection is not None
        if PYSPARK_AVAILABLE:
            assert isinstance(connection, SparkSession)

    def test_framework_connection_object_invalid_type(self, spark_session: Any) -> None:
        """Test that setting invalid connection object raises error."""
        framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        with pytest.raises(ValueError, match="Expected a SparkSession object"):
            framework.set_framework_connection_object("invalid")

    def test_transform_empty_dict(self, spark_session: Any) -> None:
        """Test transformation of empty dictionary."""
        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        result = spark_framework.transform({}, set())
        assert result is not None
        assert result.count() == 0

    def test_add_column_preserves_existing_user_row_num_column(self, spark_session: Any) -> None:
        """add_column must not clobber a pre-existing user column literally named ``__row_num``.

        The add-column path uses an internal row-number helper column to align new values with
        existing rows. A DataFrame that already owns ``__row_num`` must keep it intact alongside
        the newly added feature column.
        """
        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        existing = spark_session.createDataFrame(
            [
                {"__row_num": 100, "other": "a"},
                {"__row_num": 200, "other": "b"},
                {"__row_num": 300, "other": "c"},
            ]
        )
        spark_framework.data = existing

        result = spark_framework.transform([10, 20, 30], {"new_feature"})
        rows = result.collect()

        by_other = {row["other"]: row for row in rows}
        assert set(by_other.keys()) == {"a", "b", "c"}
        # (a) the new column exists with the expected values, aligned per original row
        assert by_other["a"]["new_feature"] == 10
        assert by_other["b"]["new_feature"] == 20
        assert by_other["c"]["new_feature"] == 30
        # (b) the original __row_num column survives with its original distinctive values
        assert "__row_num" in result.columns
        assert sorted(row["__row_num"] for row in rows) == [100, 200, 300]
        assert {by_other["a"]["__row_num"], by_other["b"]["__row_num"], by_other["c"]["__row_num"]} == {100, 200, 300}

    def test_infer_spark_type(self, spark_session: Any) -> None:
        """Test Spark type inference."""
        from pyspark.sql.types import BooleanType, IntegerType, DoubleType, StringType

        spark_framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        assert isinstance(spark_framework._infer_spark_type(True), BooleanType)
        assert isinstance(spark_framework._infer_spark_type(42), IntegerType)
        assert isinstance(spark_framework._infer_spark_type(3.14), DoubleType)
        assert isinstance(spark_framework._infer_spark_type("test"), StringType)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkDtypeExtraction(DtypeExtractionTestMixin):
    """Test SparkFramework._extract_column_dtype using shared mixin."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def dtype_sample_data(self, spark_session: Any) -> Any:
        data = [
            {"int_col": 1, "str_col": "a", "float_col": 1.0},
            {"int_col": 2, "str_col": "b", "float_col": 2.0},
            {"int_col": 3, "str_col": "c", "float_col": 3.0},
        ]
        return spark_session.createDataFrame(data)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkDataTypeValidator(DataTypeValidatorFrameworkTestMixin):
    """Test DataTypeValidator enforcement on SparkFramework using shared mixin.

    Spark exposes only one TimestampType (microsecond precision). The millisecond-
    precision tests inherited from the mixin are skipped here because Spark cannot
    express a TIMESTAMP_MILLIS column in its native schema.
    """

    @pytest.fixture
    def framework_instance(self) -> Any:
        return SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def validator_sample_data(self, spark_session: Any) -> Any:
        return self._build_spark(self.VALIDATOR_COLUMNS, spark_session)

    @pytest.fixture
    def precision_sample_data(self, spark_session: Any) -> Any:
        return self._build_spark(self.PRECISION_COLUMNS, spark_session)

    def _build_spark(self, columns: tuple[ColumnSpec, ...], spark_session: Any) -> Any:
        # Spark needs an explicit StructType, so the schema is built per-column from
        # _SPARK_TYPE_MAP. Values flow via the mixin's arrow table -> pylist.
        usable = tuple(c for c in columns if c.data_type in _SPARK_TYPE_MAP)
        schema = StructType([StructField(c.name, _SPARK_TYPE_MAP[c.data_type]) for c in usable])
        return spark_session.createDataFrame(self._arrow_table(usable).to_pylist(), schema=schema)

    def test_timestamp_ms_column_strict_ms_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Spark has only one TimestampType (microseconds); millisecond cannot be expressed")

    def test_timestamp_us_column_strict_ms_raises(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Spark has only one TimestampType (microseconds); millisecond cannot be expressed")


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkEmptyResult(EmptyResultFrameworkTestMixin):
    """Test SparkFramework._is_empty using shared mixin.

    A Spark DataFrame has no rows-without-data form: the empty case needs an explicit
    ``StructType`` so the (zero-row) frame still carries a column, mirroring how the Spark
    framework builds empty frames.
    """

    @pytest.fixture
    def framework_instance(self) -> Any:
        return SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def empty_data(self, spark_session: Any) -> Any:
        schema = StructType([StructField("a", LongType())])
        return spark_session.createDataFrame([], schema)

    @pytest.fixture
    def non_empty_data(self, spark_session: Any) -> Any:
        return spark_session.createDataFrame([{"a": 1}])


from tests.test_plugins.compute_framework.base_implementations.tfs_connection_test_mixin import TfsConnectionInitMixin  # noqa: E402


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkTfsConnectionInit(TfsConnectionInitMixin):
    @pytest.fixture
    def framework_class(self) -> Any:
        return SparkFramework

    @pytest.fixture
    def valid_connection(self, spark_session: Any) -> Any:
        return spark_session

    @pytest.fixture
    def second_valid_connection(self) -> Any:
        from unittest.mock import Mock

        return Mock(spec=SparkSession)
