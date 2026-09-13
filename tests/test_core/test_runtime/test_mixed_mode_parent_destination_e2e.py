"""E2E: mixed-mode runs must hand off data correctly across the parent/worker boundary in both
directions: a worker-owned source feeding a parent-resident destination, and a parent-resident
join result feeding its own parent-resident child, without leaking a stale or wrong-typed cfw."""

from typing import Any

import pytest

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
    import pyarrow.compute as pc
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]
    pc = None

from mloda.user import (
    mloda,
    Feature,
    DataAccessCollection,
    Index,
    JoinSpec,
    Link,
    ParallelizationMode,
    PluginCollector,
    FeatureName,
    Options,
)
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet, BaseInputData, DataCreator, MatchData
from mloda.core.runtime.flight.flight_server import FlightServer
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.runtime.compute_framework_executor import ComputeFrameworkExecutor
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from tests.test_plugins.compute_framework.base_implementations.tfs_connection_e2e_mixin import (
    TfsRawValPyArrowSource,
)


def _flight_table_keys(location: str | None) -> set[str]:
    """Location is None until the first test starts the shared session-scoped flight server process."""
    if location is None:
        return set()
    raw = FlightServer.list_flight_infos(location)
    return {key.decode("utf-8") if isinstance(key, bytes) else key for key in raw}


@pytest.fixture(autouse=True)
def _clean_flight_server(flight_server: Any) -> Any:
    """Fails a test that leaves a new table on the shared session-scoped flight server, then sweeps it."""
    before = _flight_table_keys(flight_server.location)
    yield
    after = _flight_table_keys(flight_server.location)
    try:
        assert not after - before, f"test leaked flight-server table(s): {after - before}"
    finally:
        if after:
            FlightServer.drop_tables(flight_server.location, after)


class _MixedModeDoubledDuckDBFG(FeatureGroup):
    """Destination on a SYNC-only framework; not MatchData, so no live connection lands in Options."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {DuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.project("*, raw_val * 2 AS mixed_mode_doubled")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mixed_mode_doubled"}


def _extract_mixed_mode_doubled(final: Any) -> list[int]:
    if pa is not None and isinstance(final, pa.Table):
        return list(final.column("mixed_mode_doubled").to_pylist())
    rows = final.project("mixed_mode_doubled").fetchall()
    return [row[0] for row in rows]


class _MixedModeMatchDataDuckDBFG(FeatureGroup, MatchData):
    """Destination on a SYNC/THREADING-only framework via MatchData: reproduces the leak where the
    live DuckDB connection MatchData stashes under the class-name key gets forwarded into the
    worker-eligible PyArrow source feature's pickled Options."""

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
        framework_connection_object: Any | None = None,
    ) -> Any:
        if feature_name not in cls.feature_names_supported():
            return None

        if isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            return framework_connection_object

        if data_access_collection is None:
            return None

        if data_access_collection.connections:
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
        return data.project("*, raw_val * 2 AS mixed_mode_matchdata_doubled")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mixed_mode_matchdata_doubled"}


def _extract_mixed_mode_matchdata_doubled(final: Any) -> list[int]:
    if pa is not None and isinstance(final, pa.Table):
        return list(final.column("mixed_mode_matchdata_doubled").to_pylist())
    rows = final.project("mixed_mode_matchdata_doubled").fetchall()
    return [row[0] for row in rows]


class _ThreadingOnlyPythonDictFramework(PythonDictFramework):
    """A PythonDictFramework whose steps can only ever run under SYNC or THREADING."""

    @classmethod
    def supported_parallelization_modes(cls) -> set[ParallelizationMode]:
        return {ParallelizationMode.SYNC, ParallelizationMode.THREADING}


class _MixedModeRootPythonDictFG(FeatureGroup):
    """Root source on a THREADING-only framework; parent-resident under mixed mode."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"mixed_mode_root_val"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mixed_mode_root_val": [1, 2, 3]}


class _MixedModeRootDoubledPyArrowFG(FeatureGroup):
    """Doubles the parent-resident root feature on PyArrowTable, forcing a worker-side transform."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mixed_mode_root_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("mixed_mode_root_doubled", pc.multiply(data["mixed_mode_root_val"], 2))

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mixed_mode_root_doubled"}


def _find_root_and_doubled(results: list[Any]) -> tuple[Any, Any]:
    """Classify the two results by type: dict is the root, pa.Table is the doubled feature."""
    root = next((r for r in results if isinstance(r, dict)), None)
    doubled = next((r for r in results if isinstance(r, pa.Table)), None)
    if root is None or doubled is None:
        raise AssertionError(f"expected one dict and one pa.Table result, got {[type(r) for r in results]}")
    return root, doubled


class _MixedModeDoubledThreadingPythonDictFG(FeatureGroup):
    """Destination candidate is THREADING-only or plain PythonDictFramework, whichever the run selects."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework, PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mixed_mode_doubled_threading": [v * 2 for v in data["raw_val"]]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mixed_mode_doubled_threading"}


def _extract_mixed_mode_doubled_threading(final: Any) -> list[int]:
    if pa is not None and isinstance(final, pa.Table):
        return list(final.column("mixed_mode_doubled_threading").to_pylist())
    return list(final["mixed_mode_doubled_threading"])


class _JoinLeftRootFG(FeatureGroup):
    """Left join root on the THREADING-only framework, providing join_key and left_val."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"join_key", "left_val"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("join_key",))]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"join_key": [1, 2, 3], "left_val": [10, 20, 30]}


class _JoinRightRootFG(FeatureGroup):
    """Right join root on the THREADING-only framework, providing join_key and right_val."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"join_key", "right_val"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("join_key",))]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"join_key": [1, 2, 3], "right_val": [1, 2, 3]}


class _JoinChildFG(FeatureGroup):
    """Parent-resident join consumer; a pyarrow.Table here means the join upload leaked."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("left_val"), Feature("right_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        assert isinstance(data, dict)
        return {"join_sum": [left + right for left, right in zip(data["left_val"], data["right_val"])]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"join_sum"}


class _JoinLeftValPyArrowFG(FeatureGroup):
    """Doubles left_val on PyArrowTable; arms the left root cfw's flyway upload flag."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("left_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("left_val_doubled", pc.multiply(data["left_val"], 2))

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"left_val_doubled"}


@pytest.mark.timeout(30)
@pytest.mark.skipif(pa is None, reason="PyArrow not installed.")
class TestMixedModeParentDestinationE2E:
    @pytest.mark.skipif(duckdb is None, reason="DuckDB not installed.")
    def test_worker_source_feeds_parent_duckdb_destination(self, flight_server: Any) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({TfsRawValPyArrowSource, _MixedModeDoubledDuckDBFG})
        dac = DataAccessCollection(connections={duckdb.connect()})

        result = mloda.run_all(
            [Feature("mixed_mode_doubled")],
            compute_frameworks={PyArrowTable, DuckDBFramework},
            plugin_collector=plugin_collector,
            data_access_collection=dac,
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        assert result is not None
        assert len(result) == 1
        assert _extract_mixed_mode_doubled(result[0]) == [2, 4, 6]

    @pytest.mark.skipif(duckdb is None, reason="DuckDB not installed.")
    def test_worker_source_feeds_parent_matchdata_duckdb_destination(self, flight_server: Any) -> None:
        """MatchData destination FG (unlike the sibling above): the class-name key MatchData stashes
        the live DuckDB connection under must not leak into the worker-eligible PyArrow source
        feature's pickled Options, else preflight raises ValueError before the run even starts."""
        plugin_collector = PluginCollector.enabled_feature_groups({TfsRawValPyArrowSource, _MixedModeMatchDataDuckDBFG})
        dac = DataAccessCollection(connections={duckdb.connect()})

        result = mloda.run_all(
            [Feature("mixed_mode_matchdata_doubled")],
            compute_frameworks={PyArrowTable, DuckDBFramework},
            plugin_collector=plugin_collector,
            data_access_collection=dac,
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        assert result is not None
        assert len(result) == 1
        assert _extract_mixed_mode_matchdata_doubled(result[0]) == [2, 4, 6]

    @pytest.mark.parametrize(
        "destination_framework,modes",
        [
            (_ThreadingOnlyPythonDictFramework, {ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING}),
            (PythonDictFramework, {ParallelizationMode.MULTIPROCESSING}),
        ],
        ids=["threading_destination", "pure_multiprocessing"],
    )
    def test_worker_source_feeds_parent_threading_only_destination(
        self, flight_server: Any, destination_framework: type[ComputeFramework], modes: set[ParallelizationMode]
    ) -> None:
        """THREADING and pure-MULTIPROCESSING dispatch of the worker-owned-source handoff."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {TfsRawValPyArrowSource, _MixedModeDoubledThreadingPythonDictFG}
        )

        result = mloda.run_all(
            [Feature("mixed_mode_doubled_threading")],
            compute_frameworks={PyArrowTable, destination_framework},
            plugin_collector=plugin_collector,
            parallelization_modes=modes,
            flight_server=flight_server,
        )

        assert result is not None
        assert len(result) == 1
        assert _extract_mixed_mode_doubled_threading(result[0]) == [2, 4, 6]

    def test_parent_resident_root_feature_returns_native_data_under_mixed_mode(self, flight_server: Any) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({_MixedModeRootPythonDictFG})

        result = mloda.run_all(
            [Feature("mixed_mode_root_val")],
            compute_frameworks={_ThreadingOnlyPythonDictFramework},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        assert result is not None
        assert len(result) == 1
        assert not isinstance(result[0], str)
        assert isinstance(result[0], dict)
        assert list(result[0]["mixed_mode_root_val"]) == [1, 2, 3]

    def test_parent_resident_root_feature_stays_native_when_it_also_feeds_a_worker(self, flight_server: Any) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {_MixedModeRootPythonDictFG, _MixedModeRootDoubledPyArrowFG}
        )

        result = mloda.run_all(
            [Feature("mixed_mode_root_val"), Feature("mixed_mode_root_doubled")],
            compute_frameworks={_ThreadingOnlyPythonDictFramework, PyArrowTable},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        assert result is not None
        assert len(result) == 2
        root, doubled = _find_root_and_doubled(result)

        assert not isinstance(root, str), f"root feature returned as {type(root)}"
        assert not isinstance(root, pa.Table), f"root feature returned as {type(root)}"
        assert isinstance(root, dict), f"root feature returned as {type(root)}"
        assert list(root["mixed_mode_root_val"]) == [1, 2, 3]

        assert doubled is not None
        assert doubled.column("mixed_mode_root_doubled").to_pylist() == [2, 4, 6]

        leftover = FlightServer.list_flight_infos(flight_server.location)
        assert leftover == set(), f"leaked flight tables: {leftover}"

    def test_parent_resident_root_feature_stays_native_when_it_also_feeds_a_worker_under_threading(
        self, flight_server: Any
    ) -> None:
        """THREADING dispatch of the parent-resident-root-also-feeds-a-worker handoff."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {_MixedModeRootPythonDictFG, _MixedModeRootDoubledPyArrowFG}
        )

        result = mloda.run_all(
            [Feature("mixed_mode_root_val"), Feature("mixed_mode_root_doubled")],
            compute_frameworks={_ThreadingOnlyPythonDictFramework, PyArrowTable},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        assert result is not None
        assert len(result) == 2
        root, doubled = _find_root_and_doubled(result)

        assert not isinstance(root, str), f"root feature returned as {type(root)}"
        assert not isinstance(root, pa.Table), f"root feature returned as {type(root)}"
        assert isinstance(root, dict), f"root feature returned as {type(root)}"
        assert list(root["mixed_mode_root_val"]) == [1, 2, 3]

        assert doubled is not None
        assert doubled.column("mixed_mode_root_doubled").to_pylist() == [2, 4, 6]

    def test_parent_resident_join_child_receives_native_data_under_mixed_mode(self, flight_server: Any) -> None:
        """A join result consumed by its own parent-resident child must stay native, not pa.Table,
        and the join upload must not leak a flight table onto the server (#1395)."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {_JoinLeftRootFG, _JoinRightRootFG, _JoinChildFG, _JoinLeftValPyArrowFG}
        )
        link = Link.inner(
            JoinSpec(_JoinLeftRootFG, Index(("join_key",))),
            JoinSpec(_JoinRightRootFG, Index(("join_key",))),
        )

        result = mloda.run_all(
            [Feature("join_sum"), Feature("left_val_doubled")],
            compute_frameworks={_ThreadingOnlyPythonDictFramework, PyArrowTable},
            plugin_collector=plugin_collector,
            links={link},
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        assert result is not None
        assert len(result) == 2
        join_result = next((r for r in result if isinstance(r, dict)), None)
        doubled_result = next((r for r in result if isinstance(r, pa.Table)), None)
        if join_result is None or doubled_result is None:
            raise AssertionError(f"expected one dict and one pa.Table result, got {[type(r) for r in result]}")

        assert isinstance(join_result, dict), f"join result returned as {type(join_result)}"
        assert join_result["join_sum"] == [11, 22, 33]

        assert doubled_result.column("left_val_doubled").to_pylist() == [20, 40, 60]

        leftover = FlightServer.list_flight_infos(flight_server.location)
        assert leftover == set(), f"leaked flight tables: {leftover}"


class _MpTransformSourceFG(FeatureGroup):
    """Root source on PythonDictFramework, computed in a MULTIPROCESSING-only run."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"mp_transform_source_val"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mp_transform_source_val": [1, 2, 3]}


class _MpTransformDestFG(FeatureGroup):
    """Final requested feature on PyArrowTable, reached only via a TransformFrameworkStep hop."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mp_transform_source_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("mp_transform_doubled", pc.multiply(data["mp_transform_source_val"], 2))

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mp_transform_doubled"}


@pytest.mark.timeout(30)
@pytest.mark.skipif(pa is None, reason="PyArrow not installed.")
class TestPureMultiprocessingTransformHopDoesNotLeakFlightTable:
    def test_pure_multiprocessing_transform_hop_does_not_leak_flight_table(self, flight_server: Any) -> None:
        """A source cfw whose last consumer is a TransformFrameworkStep must not leak under pure
        MULTIPROCESSING (#1395)."""
        plugin_collector = PluginCollector.enabled_feature_groups({_MpTransformSourceFG, _MpTransformDestFG})

        result = mloda.run_all(
            [Feature("mp_transform_doubled")],
            compute_frameworks={PythonDictFramework, PyArrowTable},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        assert result is not None
        assert len(result) == 1
        assert result[0].column("mp_transform_doubled").to_pylist() == [2, 4, 6]

        leftover = FlightServer.list_flight_infos(flight_server.location)
        assert leftover == set(), f"leaked flight tables: {leftover}"


class _ThreadingOnlyPyArrowTable(PyArrowTable):
    @classmethod
    def supported_parallelization_modes(cls) -> set[ParallelizationMode]:
        return {ParallelizationMode.SYNC, ParallelizationMode.THREADING}


class _XFwJoinLeftRootFG(FeatureGroup):
    """Join root on the THREADING-only framework: the source side needing the cross-framework hop."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"xfw_join_key", "xfw_left_val"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("xfw_join_key",))]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"xfw_join_key": [1, 2, 3], "xfw_left_val": [10, 20, 30]}


class _XFwJoinRightRootFG(FeatureGroup):
    """Join root on a DIFFERENT framework than the left side, forcing a TransformFrameworkStep hop."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"xfw_join_key", "xfw_right_val"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("xfw_join_key",))]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"xfw_join_key": [1, 2, 3], "xfw_right_val": [1, 2, 3]})


class _XFwJoinChildFG(FeatureGroup):
    """Sole consumer of the cross-framework join; the only feature requested from the run."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("xfw_left_val"), Feature("xfw_right_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("xfw_join_sum", pc.add(data["xfw_left_val"], data["xfw_right_val"]))

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"xfw_join_sum"}


class _CrossFwHopLeftRootFG(FeatureGroup):
    """Join root on the THREADING-only framework: the side a cross-framework join must hop out of."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"cfw_hop_join_key", "cfw_hop_left_val"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("cfw_hop_join_key",))]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"cfw_hop_join_key": [1, 2, 3], "cfw_hop_left_val": [10, 20, 30]}


class _CrossFwHopRightRootFG(FeatureGroup):
    """Join root on a DIFFERENT framework than the left side, forcing a TransformFrameworkStep hop."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"cfw_hop_join_key", "cfw_hop_right_val"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("cfw_hop_join_key",))]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"cfw_hop_join_key": [1, 2, 3], "cfw_hop_right_val": [1, 2, 3]})


class _CrossFwHopJoinChildFG(FeatureGroup):
    """Consumes both join sides, driving the cross-framework merge to completion."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("cfw_hop_left_val"), Feature("cfw_hop_right_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("cfw_hop_join_sum", pc.add(data["cfw_hop_left_val"], data["cfw_hop_right_val"]))

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"cfw_hop_join_sum"}


class _CrossFwHopSiblingFG(FeatureGroup):
    """Independent consumer of the join's source-side root feature; unrelated to the join itself."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("cfw_hop_left_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"cfw_hop_sibling_doubled": [v * 2 for v in data["cfw_hop_left_val"]]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"cfw_hop_sibling_doubled"}


@pytest.mark.timeout(30)
@pytest.mark.skipif(pa is None, reason="PyArrow not installed.")
class TestCrossFrameworkJoinHopDropTiming:
    """Regression coverage for the cross-framework branch of `_drop_join_source_if_possible`."""

    def test_cross_framework_join_hop_dropped_mid_run_not_only_at_finalize(
        self, flight_server: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The join's transported hop table must be dropped as soon as the join that consumes it
        finishes, not only by the blanket finalize sweep. A `TransformFrameworkStep.execute` spy
        identifies the hop's own cfw uuid (the flight-table key to watch): there is no public-API
        way to name it, since `PlanStep` records no uuid for "transform" steps."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {_XFwJoinLeftRootFG, _XFwJoinRightRootFG, _XFwJoinChildFG}
        )
        link = Link.inner(
            JoinSpec(_XFwJoinLeftRootFG, Index(("xfw_join_key",))),
            JoinSpec(_XFwJoinRightRootFG, Index(("xfw_join_key",))),
        )

        hop_uuids: list[str] = []
        original_execute = TransformFrameworkStep.execute

        def _spy_execute(self: Any, cfw_register: Any, cfw: Any, from_cfw: Any = None, data: Any = None) -> Any:
            hop_uuids.append(str(cfw.uuid))
            return original_execute(self, cfw_register, cfw, from_cfw=from_cfw, data=data)

        monkeypatch.setattr(TransformFrameworkStep, "execute", _spy_execute)

        stream = mloda.stream_all(
            [Feature("xfw_join_sum")],
            compute_frameworks={_ThreadingOnlyPythonDictFramework, _ThreadingOnlyPyArrowTable},
            plugin_collector=plugin_collector,
            links={link},
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        join_steps = [step for step in stream.plan if step.step_kind == "join"]
        assert len(join_steps) == 1
        assert join_steps[0].compute_framework != join_steps[0].source_compute_framework, (
            "fixture must produce a cross-framework JoinStep (destination_framework != source_framework)"
        )

        result = next(stream)
        assert result.column("xfw_join_sum").to_pylist() == [11, 22, 33]
        assert len(hop_uuids) == 1, f"expected exactly one TransformFrameworkStep hop, got {hop_uuids}"

        # Drain and finalize in `finally` regardless of the assertion's outcome: a failure here must
        # not also leave the run's own cleanup undone, which would otherwise fail _clean_flight_server's
        # own leak check with unrelated noise on top of the real assertion below.
        try:
            mid_run_keys = _flight_table_keys(flight_server.location)
            assert hop_uuids[0] not in mid_run_keys, (
                f"cross-framework join hop {hop_uuids[0]} is still on the flight server right after its "
                f"own join finished: {mid_run_keys}; it should have been dropped incrementally, not left "
                "for the run-finalize sweep"
            )
        finally:
            remaining = list(stream)

        assert remaining == []

        leftover = FlightServer.list_flight_infos(flight_server.location)
        assert leftover == set(), f"leaked flight tables: {leftover}"

    def test_shared_cross_framework_hop_source_root_survives_until_join_and_sibling_finish(
        self, flight_server: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guards the join's SOURCE ROOT cfw (not the hop itself, already covered above): it also
        feeds an independent sibling feature, so it must survive until BOTH the sibling and the
        join have consumed it, not just the first to finish. A `ComputeFramework.upload_finished_data`
        spy identifies the source root's own cfw uuid (the specific flight-table key to watch):
        asserting mere non-emptiness of the flight server would still pass under a mutation that
        force-drops this exact cfw too early, so long as anything else remains on the server."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {_CrossFwHopLeftRootFG, _CrossFwHopRightRootFG, _CrossFwHopJoinChildFG, _CrossFwHopSiblingFG}
        )
        link = Link.inner(
            JoinSpec(_CrossFwHopLeftRootFG, Index(("cfw_hop_join_key",))),
            JoinSpec(_CrossFwHopRightRootFG, Index(("cfw_hop_join_key",))),
        )

        source_root_uuids: list[str] = []
        original_upload = ComputeFramework.upload_finished_data

        def _spy_upload(self: Any, location: str) -> Any:
            if type(self).get_class_name() == _ThreadingOnlyPythonDictFramework.get_class_name():
                source_root_uuids.append(str(self.uuid))
            return original_upload(self, location)

        monkeypatch.setattr(ComputeFramework, "upload_finished_data", _spy_upload)

        stream = mloda.stream_all(
            [Feature("cfw_hop_join_sum"), Feature("cfw_hop_sibling_doubled")],
            compute_frameworks={_ThreadingOnlyPythonDictFramework, _ThreadingOnlyPyArrowTable},
            plugin_collector=plugin_collector,
            links={link},
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        join_steps = [step for step in stream.plan if step.step_kind == "join"]
        assert len(join_steps) == 1
        assert join_steps[0].compute_framework != join_steps[0].source_compute_framework, (
            "fixture must produce a cross-framework JoinStep (destination_framework != source_framework)"
        )

        frames = stream.frames()

        first_step, first_result = next(frames)
        assert first_step.feature_names == ("cfw_hop_sibling_doubled",), (
            "expected the independent sibling to finish before the cross-framework join's own child"
        )
        assert list(first_result["cfw_hop_sibling_doubled"]) == [20, 40, 60]

        assert len(source_root_uuids) == 1, f"expected exactly one source-root upload, got {source_root_uuids}"
        mid_run_keys = _flight_table_keys(flight_server.location)
        assert source_root_uuids[0] in mid_run_keys, (
            f"the join has not finished yet: source root {source_root_uuids[0]} must still be on "
            f"the flight server, not dropped just because the unrelated sibling finished first: {mid_run_keys}"
        )

        remaining = list(frames)
        assert len(remaining) == 1
        second_step, second_result = remaining[0]
        assert second_step.feature_names == ("cfw_hop_join_sum",)
        assert second_result.column("cfw_hop_join_sum").to_pylist() == [11, 22, 33]

        leftover = FlightServer.list_flight_infos(flight_server.location)
        assert leftover == set(), f"leaked flight tables: {leftover}"


class _Gap1SharedRootFG(FeatureGroup):
    """Shared root feeding both a cross-framework hop and an independent same-framework chain."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"gap1_shared_val"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"gap1_shared_val": [1, 2, 3]}


class _Gap1HopDestFG(FeatureGroup):
    """Plain (non-join) cross-framework hop consumer of the shared root."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("gap1_shared_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("gap1_hop_doubled", pc.multiply(data["gap1_shared_val"], 2))

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"gap1_hop_doubled"}


class _Gap1SiblingGate1FG(FeatureGroup):
    """First of two same-framework gates ahead of the independent sibling: pads its own chain
    long enough that it cannot finish within the same pass as the (much shorter) hop branch."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("gap1_shared_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {**data, "gap1_sibling_gate1_val": list(data["gap1_shared_val"])}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"gap1_sibling_gate1_val"}


class _Gap1SiblingGate2FG(FeatureGroup):
    """Second gate; see `_Gap1SiblingGate1FG`."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("gap1_sibling_gate1_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {**data, "gap1_sibling_gate2_val": list(data["gap1_sibling_gate1_val"])}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"gap1_sibling_gate2_val"}


class _Gap1SiblingFG(FeatureGroup):
    """Independent, same-framework consumer of the shared root, delayed by two gates."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("gap1_sibling_gate2_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"gap1_sibling_tripled": [v * 3 for v in data["gap1_sibling_gate2_val"]]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"gap1_sibling_tripled"}


class _DiamondHopRootFG(FeatureGroup):
    """Root feature feeding both a plain cross-framework hop and a same-framework diamond
    descendant that also reads the hop's own output."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"diamond_hop_root_val"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"diamond_hop_root_val": [1, 2, 3]}


class _DiamondHopDestFG(FeatureGroup):
    """Plain (non-join) cross-framework hop consumer of the root."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("diamond_hop_root_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("diamond_hop_doubled", pc.multiply(data["diamond_hop_root_val"], 2))

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"diamond_hop_doubled"}


class _DiamondHopDescendantFG(FeatureGroup):
    """Reads the root DIRECTLY (same framework as the root, not via the hop) as well as the hop's
    own output feature: a graph-descendant of BOTH the hop's source and its consumer, which
    `owed_tokens` must not let the hop's own finish drop early."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("diamond_hop_root_val"), Feature("diamond_hop_doubled")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"diamond_hop_result": [v * 7 for v in data["diamond_hop_root_val"]]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"diamond_hop_result"}


@pytest.mark.timeout(30)
@pytest.mark.skipif(pa is None, reason="PyArrow not installed.")
class TestTransformFrameworkStepSourceRootDropTiming:
    """Regression coverage for a TransformFrameworkStep's FROM-side (source) cfw: unlike a
    JoinStep, `ExecutionOrchestrator._process_step_result` never marks anything as arrived on the
    cfw a finished TransformFrameworkStep just consumed, so it survives until the run-finalize
    sweep instead of being dropped incrementally once the hop that reads it has finished."""

    def test_plain_hop_source_root_dropped_mid_run_not_only_at_finalize(
        self, flight_server: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The simplest possible shape: one MULTIPROCESSING root, one plain (non-join) hop, one
        requested feature. A `ComputeFrameworkExecutor.add_compute_framework` spy identifies the
        source root's own cfw uuid (the flight-table key to watch): there is no public-API way to
        name it directly, mirroring the `TransformFrameworkStep.execute` spy used above for the
        hop's own uuid."""
        plugin_collector = PluginCollector.enabled_feature_groups({_MpTransformSourceFG, _MpTransformDestFG})

        source_root_uuids: list[str] = []
        original_add_cfw = ComputeFrameworkExecutor.add_compute_framework

        def _spy_add_cfw(
            self: Any, step: Any, parallelization_mode: Any, feature_uuid: Any, children_if_root: Any
        ) -> Any:
            result_uuid = original_add_cfw(self, step, parallelization_mode, feature_uuid, children_if_root)
            if step.feature_group is _MpTransformSourceFG:
                source_root_uuids.append(str(result_uuid))
            return result_uuid

        monkeypatch.setattr(ComputeFrameworkExecutor, "add_compute_framework", _spy_add_cfw)

        stream = mloda.stream_all(
            [Feature("mp_transform_doubled")],
            compute_frameworks={PythonDictFramework, PyArrowTable},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        result = next(stream)
        assert result.column("mp_transform_doubled").to_pylist() == [2, 4, 6]
        assert len(source_root_uuids) == 1, f"expected exactly one source-root cfw, got {source_root_uuids}"

        # Drain and finalize in `finally` regardless of the assertion's outcome: see the
        # analogous comment on the cross-framework join hop test above.
        try:
            mid_run_keys = _flight_table_keys(flight_server.location)
            assert source_root_uuids[0] not in mid_run_keys, (
                f"TransformFrameworkStep source root {source_root_uuids[0]} is still on the flight "
                f"server right after its own hop finished: {mid_run_keys}; it should have been "
                "dropped incrementally, not left for the run-finalize sweep"
            )
        finally:
            remaining = list(stream)

        assert remaining == []

        leftover = FlightServer.list_flight_infos(flight_server.location)
        assert leftover == set(), f"leaked flight tables: {leftover}"

    def test_shared_root_survives_hop_until_independent_sibling_also_finishes(
        self, flight_server: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shared root also feeds an independent, same-framework sibling chain (delayed by
        two gates so it cannot race the much shorter hop branch): the root must survive until
        BOTH have consumed it. Allowed to currently pass (today's bug under-drops via the
        finalize-only sweep; it never over-drops). A `ComputeFramework.upload_finished_data` spy
        identifies the shared root's own cfw uuid (the specific flight-table key to watch):
        asserting mere non-emptiness of the flight server would still pass under a mutation that
        force-drops this exact cfw too early, so long as anything else remains on the server."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                _Gap1SharedRootFG,
                _Gap1HopDestFG,
                _Gap1SiblingGate1FG,
                _Gap1SiblingGate2FG,
                _Gap1SiblingFG,
            }
        )

        shared_root_uuids: list[str] = []
        original_upload = ComputeFramework.upload_finished_data

        def _spy_upload(self: Any, location: str) -> Any:
            if type(self).get_class_name() == _ThreadingOnlyPythonDictFramework.get_class_name():
                shared_root_uuids.append(str(self.uuid))
            return original_upload(self, location)

        monkeypatch.setattr(ComputeFramework, "upload_finished_data", _spy_upload)

        stream = mloda.stream_all(
            [Feature("gap1_hop_doubled"), Feature("gap1_sibling_tripled")],
            compute_frameworks={_ThreadingOnlyPythonDictFramework, _ThreadingOnlyPyArrowTable},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        frames = stream.frames()

        first_step, first_result = next(frames)
        assert first_step.feature_names == ("gap1_hop_doubled",), (
            "expected the (structurally much shorter) hop branch to finish before the "
            "deliberately delayed independent sibling"
        )
        assert first_result.column("gap1_hop_doubled").to_pylist() == [2, 4, 6]

        assert len(shared_root_uuids) == 1, f"expected exactly one shared-root upload, got {shared_root_uuids}"
        mid_run_keys = _flight_table_keys(flight_server.location)
        assert shared_root_uuids[0] in mid_run_keys, (
            f"the independent sibling has not finished yet: the shared root's own upload "
            f"{shared_root_uuids[0]} must still be on the flight server, not dropped just because "
            f"the hop alone finished: {mid_run_keys}"
        )

        second_step, second_result = next(frames)
        assert second_step.feature_names == ("gap1_sibling_tripled",)
        assert list(second_result["gap1_sibling_tripled"]) == [3, 6, 9]

        remaining = list(frames)
        assert remaining == []

        leftover = FlightServer.list_flight_infos(flight_server.location)
        assert leftover == set(), f"leaked flight tables: {leftover}"

    def test_diamond_descendant_of_hop_source_and_consumer_survives_hop_finish(self, flight_server: Any) -> None:
        """A feature that graph-descends from BOTH the hop's source root AND the hop's own
        consumer output must not be starved of the root: crediting the hop's consumer's full
        `children_if_root` (instead of just the consumer's own uuids) into `owed_tokens`
        double-counts this descendant, so the root's `children_if_root` looks fully satisfied and
        gets dropped as soon as the hop finishes, even though this descendant's own step still
        needs to read the root directly. Pure SYNC: the drop mechanism runs identically in-process,
        so no flight server or MULTIPROCESSING is needed to reproduce the bug."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {_DiamondHopRootFG, _DiamondHopDestFG, _DiamondHopDescendantFG}
        )

        result = mloda.run_all(
            [Feature("diamond_hop_result")],
            compute_frameworks={PythonDictFramework, PyArrowTable},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert result is not None
        assert len(result) == 1
        assert list(result[0]["diamond_hop_result"]) == [7, 14, 21]
