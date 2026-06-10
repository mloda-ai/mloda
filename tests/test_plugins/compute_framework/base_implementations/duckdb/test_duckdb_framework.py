import os
from datetime import datetime, timezone
from typing import Any, Optional
import pytest
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda.user import FeatureName
from mloda.user import ParallelizationMode
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

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment]


class TestDuckDBFrameworkAvailability:
    def test_is_available_when_duckdb_not_installed(self) -> None:
        """Test that is_available() returns False when duckdb import fails."""
        assert_unavailable_when_import_blocked(DuckDBFramework, ["duckdb"])


class TestDuckDBInstallation:
    @pytest.mark.skipif(
        os.getenv("SKIP_DUCKDB_INSTALLATION_TEST", "false").lower() == "true",
        reason="DuckDB installation test is disabled by environment variable",
    )
    def test_duckdb_is_installed(self) -> None:
        """Test that DuckDB is properly installed and can be imported."""
        try:
            import duckdb
            import pyarrow as pa

            # Test basic functionality
            conn = duckdb.connect()
            data = {"a": [1, 2, 3], "b": [4, 5, 6]}
            arrow_table = pa.Table.from_pydict(data)
            relation = conn.from_arrow(arrow_table)
            result = relation.df()
            assert len(result) == 3
            assert list(result.columns) == ["a", "b"]
        except ImportError:
            pytest.fail("DuckDB is not installed but is required for this test environment")


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed. Skipping this test.")
class TestDuckDBFrameworkComputeFramework:
    @pytest.fixture
    def duckdb_framework(self) -> DuckDBFramework:
        """Create a fresh DuckDBFramework instance for each test."""
        return DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def expected_data(self, connection: Any, dict_data: dict[str, list[int]]) -> Any:
        """Create fresh expected DuckDB relation for each test."""
        expected_arrow = pa.Table.from_pydict(dict_data)
        return DuckdbRelation.from_arrow(connection, expected_arrow)

    def test_expected_data_framework(self, duckdb_framework: DuckDBFramework) -> None:
        assert duckdb_framework.expected_data_framework() == DuckdbRelation

    def test_transform_dict_to_relation(
        self, duckdb_framework: DuckDBFramework, connection: Any, dict_data: dict[str, list[int]], expected_data: Any
    ) -> None:
        duckdb_framework.set_framework_connection_object(connection)
        result = duckdb_framework.transform(dict_data, set())
        result_df = result.df()
        expected_df = expected_data.df()

        # Compare the dataframes
        assert result_df.equals(expected_df)

    def test_transform_invalid_data(self, duckdb_framework: DuckDBFramework) -> None:
        with pytest.raises(ValueError):
            duckdb_framework.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self, duckdb_framework: DuckDBFramework, expected_data: Any) -> None:
        data = duckdb_framework.select_data_by_column_names(expected_data, {FeatureName("column1")})
        assert isinstance(data, pa.Table)
        assert "column1" in data.column_names

    def test_set_column_names(self, duckdb_framework: DuckDBFramework, expected_data: Any) -> None:
        duckdb_framework.data = expected_data
        duckdb_framework.set_column_names()
        assert "column1" in duckdb_framework.column_names
        assert "column2" in duckdb_framework.column_names

    def test_transform_add_column_preserves_existing(self, duckdb_framework: DuckDBFramework, connection: Any) -> None:
        duckdb_framework.set_framework_connection_object(connection)
        dict_data = {"col_a": [1, 2, 3], "col_b": [4, 5, 6]}
        duckdb_framework.data = duckdb_framework.transform(dict_data, set())
        result = duckdb_framework.transform(data=[7, 8, 9], feature_names={"col_c"})
        assert set(result.columns) == {"col_a", "col_b", "col_c"}
        assert len(result) == 3
        arrow = result.to_arrow_table()
        assert arrow.column("col_a").to_pylist() == [1, 2, 3]
        assert arrow.column("col_c").to_pylist() == [7, 8, 9]


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed. Skipping this test.")
class TestDuckDBFrameworkSessionTimezone:
    """Timestamp flooring must be UTC-based regardless of the connection's preset session timezone.

    DuckDB's ``DATE_TRUNC`` on a ``TIMESTAMPTZ`` operates in the connection's session
    timezone. ``set_framework_connection_object`` is the contract chokepoint: after a
    user-supplied connection is handed to the framework, flooring a known UTC instant to
    the day must yield the UTC-floored instant, NOT the local-midnight instant of whatever
    timezone the connection happened to be preset to (issues #522/#523).

    The chosen instant ``2023-01-01 02:00:00+00`` floors (to day) to:
      - UTC:                 2023-01-01 00:00:00+00  (expected)
      - America/New_York:    2022-12-31 05:00:00+00  (leaked-tz bug)
      - Asia/Kolkata:        2022-12-31 18:30:00+00  (leaked-tz bug)
    so the bug is unambiguously visible as a wrong absolute instant.
    """

    # A UTC instant a couple of hours past midnight: still 2023-01-01 in UTC, but rolls
    # back to the previous day once a negative-offset session tz leaks in.
    KNOWN_INSTANT = datetime(2023, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
    EXPECTED_UTC_DAY_FLOOR = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def duckdb_framework(self) -> DuckDBFramework:
        return DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    def _timestamptz_relation(self, framework: DuckDBFramework, instant: datetime) -> DuckdbRelation:
        """Build a single-row TIMESTAMPTZ relation on the framework's connection."""
        conn = framework.get_framework_connection_object()
        arrow_table = pa.table({"ts": pa.array([instant], type=pa.timestamp("us", tz="UTC"))})
        return DuckdbRelation.from_arrow(conn, arrow_table)

    @staticmethod
    def _floored_instant(relation: DuckdbRelation) -> datetime:
        """Force lazy materialization and return the floored instant normalized to UTC."""
        arrow_result = relation.to_arrow_table()
        value = arrow_result.column("ts").to_pylist()[0]
        assert isinstance(value, datetime)
        return value.astimezone(timezone.utc)

    def test_date_trunc_day_floor_is_utc_via_project(
        self, duckdb_framework: DuckDBFramework, non_utc_connection: Any, non_utc_zone: str
    ) -> None:
        """DATE_TRUNC day-flooring via the project() API must ignore the preset session tz."""
        duckdb_framework.set_framework_connection_object(non_utc_connection)
        relation = self._timestamptz_relation(duckdb_framework, self.KNOWN_INSTANT)
        floored = relation.project('DATE_TRUNC(\'day\', "ts") AS "ts"')
        result = self._floored_instant(floored)
        assert result == self.EXPECTED_UTC_DAY_FLOOR, (
            f"Day floor leaked session timezone {non_utc_zone!r}: expected {self.EXPECTED_UTC_DAY_FLOOR}, got {result}"
        )

    def test_date_trunc_day_floor_is_utc_via_query(
        self, duckdb_framework: DuckDBFramework, non_utc_connection: Any, non_utc_zone: str
    ) -> None:
        """DATE_TRUNC day-flooring via the query() API must ignore the preset session tz."""
        duckdb_framework.set_framework_connection_object(non_utc_connection)
        relation = self._timestamptz_relation(duckdb_framework, self.KNOWN_INSTANT)
        floored = relation.query("ts_rel", 'SELECT DATE_TRUNC(\'day\', "ts") AS "ts" FROM ts_rel')
        result = self._floored_instant(floored)
        assert result == self.EXPECTED_UTC_DAY_FLOOR, (
            f"Day floor leaked session timezone {non_utc_zone!r}: expected {self.EXPECTED_UTC_DAY_FLOOR}, got {result}"
        )


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed. Skipping this test.")
class TestDuckDBFrameworkMerge(DataFrameTestBase):
    """Test DuckDBFramework merge operations using the base test class."""

    @classmethod
    def framework_class(cls) -> type[Any]:
        """Return the DuckDBFramework class."""
        return DuckDBFramework

    def setup_method(self) -> None:
        """Set up DuckDB connection and test data."""
        self.conn = duckdb.connect()
        super().setup_method()

    def create_dataframe(self, data: dict[str, Any]) -> Any:
        """Create a DuckDB relation from a dictionary."""
        arrow_table = pa.Table.from_pydict(data)
        return DuckdbRelation.from_arrow(self.conn, arrow_table)

    def get_connection(self) -> Optional[Any]:
        """Return DuckDB connection object."""
        return self.conn

    def _create_test_framework(self) -> Any:
        """Create a framework instance with sync mode and DuckDB connection."""
        framework = super()._create_test_framework()
        framework.set_framework_connection_object(self.conn)
        return framework

    def _get_merge_engine(self, framework: Any) -> Any:
        """Get merge engine factory that returns an instance with connection for DuckDB."""
        merge_engine_class = framework.merge_engine()
        framework_connection = framework.get_framework_connection_object()

        class MergeEngineFactory:
            def __call__(self) -> Any:
                return merge_engine_class(framework_connection)

        return MergeEngineFactory()


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed. Skipping this test.")
class TestDuckDBDtypeExtraction(DtypeExtractionTestMixin):
    """Test DuckDBFramework._extract_column_dtype using shared mixin."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def dtype_sample_data(self, connection: Any) -> Any:
        arrow_table = pa.Table.from_pydict(
            {"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.0, 2.0, 3.0]}
        )
        return DuckdbRelation.from_arrow(connection, arrow_table)


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed. Skipping this test.")
class TestDuckDBDataTypeValidator(DataTypeValidatorFrameworkTestMixin):
    """Test DataTypeValidator enforcement on DuckDBFramework using shared mixin."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def validator_sample_data(self, connection: Any) -> Any:
        return DuckdbRelation.from_arrow(connection, self._arrow_table(self.VALIDATOR_COLUMNS))

    @pytest.fixture
    def precision_sample_data(self, connection: Any) -> Any:
        return DuckdbRelation.from_arrow(connection, self._arrow_table(self.PRECISION_COLUMNS))


from tests.test_plugins.compute_framework.base_implementations.tfs_connection_test_mixin import TfsConnectionInitMixin  # noqa: E402
from unittest.mock import patch  # noqa: E402
from mloda.user import DataAccessCollection  # noqa: E402
import mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework as duckdb_framework_module  # noqa: E402


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed.")
class TestDuckDBTfsConnectionInit(TfsConnectionInitMixin):
    @pytest.fixture
    def framework_class(self) -> Any:
        return DuckDBFramework

    @pytest.fixture
    def valid_connection(self) -> Any:
        conn = duckdb.connect()
        yield conn
        conn.close()

    @pytest.fixture
    def second_valid_connection(self) -> Any:
        conn = duckdb.connect()
        yield conn
        conn.close()

    def test_returns_none_when_duckdb_module_missing(self, framework_class: Any) -> None:
        """When the duckdb module symbol is None (optional dep missing), the classmethod
        must return None without raising AttributeError, matching Spark/Iceberg."""
        dac = DataAccessCollection(connections={object()})
        with patch.object(duckdb_framework_module, "duckdb", None):
            assert framework_class.pick_connection_from_dac(dac) is None
