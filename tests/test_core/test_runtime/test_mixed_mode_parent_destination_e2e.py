"""E2E: a MULTIPROCESSING-eligible source feeding a SYNC-only destination framework in a
mixed-mode run must not hand the destination a stale parent-side compute framework instance."""

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

from mloda.user import mloda, Feature, DataAccessCollection, ParallelizationMode, PluginCollector, FeatureName, Options
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from tests.test_plugins.compute_framework.base_implementations.tfs_connection_e2e_mixin import (
    TfsRawValPyArrowSource,
)


class _MixedModeDoubledDuckDBFG(FeatureGroup):
    """Destination on a SYNC-only framework. Plain FeatureGroup, not MatchData, so no live
    connection lands in Options."""

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


@pytest.mark.timeout(30)
@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow not installed.")
class TestMixedModeParentDestinationE2E:
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
