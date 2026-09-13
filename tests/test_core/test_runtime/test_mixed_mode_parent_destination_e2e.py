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
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet, BaseInputData, DataCreator
from mloda.core.runtime.flight.flight_server import FlightServer
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
    """Destination on a THREADING-only framework. Plain FeatureGroup on columnar dict data."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_ThreadingOnlyPythonDictFramework}

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


class _MixedModeDoubledMultiprocessingPythonDictFG(FeatureGroup):
    """Destination on plain PythonDictFramework, which supports MULTIPROCESSING unlike the THREADING-only variant."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mixed_mode_doubled_multiprocessing": [v * 2 for v in data["raw_val"]]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mixed_mode_doubled_multiprocessing"}


def _extract_mixed_mode_doubled_multiprocessing(final: Any) -> list[int]:
    if pa is not None and isinstance(final, pa.Table):
        return list(final.column("mixed_mode_doubled_multiprocessing").to_pylist())
    return list(final["mixed_mode_doubled_multiprocessing"])


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

    def test_worker_source_feeds_parent_threading_only_destination(self, flight_server: Any) -> None:
        """THREADING dispatch of the worker-owned-source handoff."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {TfsRawValPyArrowSource, _MixedModeDoubledThreadingPythonDictFG}
        )

        result = mloda.run_all(
            [Feature("mixed_mode_doubled_threading")],
            compute_frameworks={PyArrowTable, _ThreadingOnlyPythonDictFramework},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING},
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
        """A join result consumed by its own parent-resident child must stay native, not pa.Table."""
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

    def test_pure_multiprocessing_transform_step_leaves_no_flight_table(self, flight_server: Any) -> None:
        """A TFS-only MULTIPROCESSING run (no SYNC/THREADING mixed in) must not leak its upload."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {TfsRawValPyArrowSource, _MixedModeDoubledMultiprocessingPythonDictFG}
        )

        result = mloda.run_all(
            [Feature("mixed_mode_doubled_multiprocessing")],
            compute_frameworks={PyArrowTable, PythonDictFramework},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        assert result is not None
        assert len(result) == 1
        assert _extract_mixed_mode_doubled_multiprocessing(result[0]) == [2, 4, 6]
