"""End-to-end TFS connection-propagation test for DuckDB.

Reproducer for issue #440. PyArrow -> DuckDB TFS path; without the fix the
destination DuckDBFramework has no connection and `data.project(...)` raises."""

import sqlite3
from typing import Any

import pytest

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]

from mloda.user import Feature, DataAccessCollection, ParallelizationMode, PluginCollector, mloda
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet, MatchData, BaseInputData
from mloda.user import FeatureName, Options
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from tests.test_plugins.compute_framework.base_implementations.tfs_connection_e2e_mixin import (
    TfsConnectionEndToEndMixin,
    TfsRawValPyArrowSource,
    _ROOT_SEEN,
    root_binding_groups,
)


class TfsDoubledDuckDBFG(FeatureGroup, MatchData):
    """Doubles raw_val using a DuckDB SQL projection that REQUIRES the connection."""

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
        framework_connection_object: Any | None = None,
    ) -> Any:
        if duckdb is None or not DuckDBFramework.is_available():
            return None
        if feature_name not in cls.feature_names_supported():
            return None
        if isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            return framework_connection_object
        if data_access_collection is None:
            return None
        for conn in data_access_collection.connections.values():
            if isinstance(conn, duckdb.DuckDBPyConnection):
                return conn
        return None

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {DuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.project("*, raw_val * 2 AS tfs_doubled")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tfs_doubled"}


_SEEN: dict[str, Any] = {}


class DcArrowSink(FeatureGroup):
    """DuckDB-only consumer of the PyArrow source; records the connection of its transformed input."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {DuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        name = str(features.get_name_of_one_feature())
        _SEEN[name] = data.connection
        return data.project(f"raw_val * 2 AS {name}")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"dc_arrow_a", "dc_arrow_b"}


_OTHER: dict[str, Any] = {}


class DcWrongConnRoot(FeatureGroup):
    """DuckDB root returning a relation that lives on a connection other than its handle's."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"dc_wrong"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {DuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        return DuckdbRelation.from_arrow(_OTHER["conn"], pa.Table.from_pydict({"dc_wrong": [1]}))


class DcChainRoot(FeatureGroup):
    """DuckDB root; the cfw it runs on is bound to the root's own handle."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"dc_chain_src"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {DuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"dc_chain_src": [1, 2, 3]}


class DcChainSink(FeatureGroup):
    """DuckDB consumer on its own handle over a root on another one (the handle is not forwarded)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("dc_chain_src", options=Options(group={"connection_handle": "a"}), forward_group=False)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {DuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.project("dc_chain_src * 2 AS dc_chain_out")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"dc_chain_out"}


def _handle(handle: str) -> Options:
    return Options(group={"connection_handle": handle})


def _run(features: list[Feature], groups: set[Any], frameworks: list[Any], dac: DataAccessCollection) -> Any:
    _SEEN.clear()
    return mloda.run_all(
        features,
        compute_frameworks=frameworks,
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
        data_access_collection=dac,
        parallelization_modes={ParallelizationMode.SYNC},
    )


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow not installed.")
class TestDuckDBRootConnectionBinding:
    def test_an_explicit_handle_binds_the_root_to_that_connection_among_two(self) -> None:
        root, sink, dst = root_binding_groups(DuckDBFramework, TestDuckDBTfsConnectionE2E.double_relation)
        a, b = duckdb.connect(), duckdb.connect()
        dac = DataAccessCollection({"a": a, "b": b})
        result = _run([Feature(dst, options=_handle("b"))], {root, sink}, [DuckDBFramework], dac)
        assert sorted(result[0].column(dst).to_pylist()) == [2, 4, 6]
        assert _ROOT_SEEN[dst] is b

    def test_data_on_another_connection_than_the_explicit_handle_raises(self) -> None:
        a, b = duckdb.connect(), duckdb.connect()
        dac = DataAccessCollection({"a": a, "b": b})
        _OTHER["conn"] = b
        with pytest.raises(Exception, match="DuckDBFramework.*connection_handle"):
            _run([Feature("dc_wrong", options=_handle("a"))], {DcWrongConnRoot}, [DuckDBFramework], dac)

    def test_a_named_handle_picks_the_transform_connection_into_a_duckdb_consumer(self) -> None:
        a, b = duckdb.connect(), duckdb.connect()
        dac = DataAccessCollection({"a": a, "b": b})
        result = _run(
            [Feature("dc_arrow_b", options=_handle("b"))],
            {TfsRawValPyArrowSource, DcArrowSink},
            [PyArrowTable, DuckDBFramework],
            dac,
        )
        assert result[0].column("dc_arrow_b").to_pylist() == [2, 4, 6]
        assert _SEEN["dc_arrow_b"] is b

    def test_an_explicit_handle_on_a_cfw_bound_to_another_connection_raises(self) -> None:
        dac = DataAccessCollection({"a": duckdb.connect(), "b": duckdb.connect()})
        with pytest.raises(Exception, match="connection_handle"):
            _run([Feature("dc_chain_out", options=_handle("b"))], {DcChainRoot, DcChainSink}, [DuckDBFramework], dac)

    def test_converted_results_on_different_handles_fail_at_planning_naming_both(self) -> None:
        dac = DataAccessCollection({"a": duckdb.connect(), "b": duckdb.connect()})
        with pytest.raises(Exception, match="connection") as raised:
            mloda.run_all(
                [Feature("raw_val", options=_handle("a")), Feature("raw_val", options=_handle("b"))],
                compute_frameworks=[PyArrowTable, DuckDBFramework],
                plugin_collector=PluginCollector.enabled_feature_groups({TfsRawValPyArrowSource}),
                data_access_collection=dac,
                parallelization_modes={ParallelizationMode.SYNC},
                output_framework=DuckDBFramework,
            )
        assert "'a'" in str(raised.value)
        assert "'b'" in str(raised.value)

    def test_a_sqlite_handle_does_not_block_the_duckdb_output_connection(self) -> None:
        root, sink, dst = root_binding_groups(
            SqliteFramework, lambda data, src, out: data.select(_raw_sql=f"*, {src} * 2 AS {out}")
        )
        d = duckdb.connect()
        dac = DataAccessCollection({"s": sqlite3.connect(":memory:"), "d": d})
        result = mloda.run_all(
            [Feature(dst, options=_handle("s"))],
            compute_frameworks=[SqliteFramework, DuckDBFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({root, sink}),
            data_access_collection=dac,
            parallelization_modes={ParallelizationMode.SYNC},
            output_framework=DuckDBFramework,
        )
        assert result[0].connection is d
        assert sorted(result[0].to_arrow_table().column(dst).to_pylist()) == [2, 4, 6]

    def test_a_hop_into_duckdb_with_a_foreign_handle_reports_the_resolver_reason(self) -> None:
        dac = DataAccessCollection({"s": sqlite3.connect(":memory:"), "a": duckdb.connect(), "b": duckdb.connect()})
        with pytest.raises(Exception, match="does not accept") as raised:
            _run(
                [Feature("dc_arrow_a", options=_handle("s"))],
                {TfsRawValPyArrowSource, DcArrowSink},
                [PyArrowTable, DuckDBFramework],
                dac,
            )
        assert "'s'" in str(raised.value)
        assert "DuckDBFramework: Connection handle 's' holds a connection this framework does not accept." in str(
            raised.value
        )

    def test_consumers_on_different_handles_never_share_one_transform_hop(self) -> None:
        a, b = duckdb.connect(), duckdb.connect()
        dac = DataAccessCollection({"a": a, "b": b})
        result = _run(
            [Feature("dc_arrow_a", options=_handle("a")), Feature("dc_arrow_b", options=_handle("b"))],
            {TfsRawValPyArrowSource, DcArrowSink},
            [PyArrowTable, DuckDBFramework],
            dac,
        )
        assert len(result) == 2
        assert _SEEN["dc_arrow_a"] is a
        assert _SEEN["dc_arrow_b"] is b


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow not installed.")
class TestDuckDBTfsConnectionE2E(TfsConnectionEndToEndMixin):
    destination_framework_class = DuckDBFramework
    destination_fg_class = TfsDoubledDuckDBFG

    @staticmethod
    def double_relation(data: Any, src: str, dst: str) -> Any:
        return data.project(f"{src} * 2 AS {dst}")

    @pytest.fixture
    def live_connection(self) -> Any:
        conn = duckdb.connect()
        yield conn
        conn.close()

    def extract_doubled(self, final: Any) -> list[int]:
        assert isinstance(final, pa.Table)
        assert "tfs_doubled" in final.column_names
        return list(final.column("tfs_doubled").to_pylist())
