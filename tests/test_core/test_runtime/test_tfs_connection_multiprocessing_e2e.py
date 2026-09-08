"""E2E: a ConnectionSpec registered in DataAccessCollection lets a MULTIPROCESSING worker (and,
when the destination is the final requested feature, the parent) open its own connection, while a
live connection is rejected before any worker starts."""

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

from mloda.user import ConnectionSpec, Feature, DataAccessCollection, ParallelizationMode, PluginCollector, mloda
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class _MultiprocessingConnectionDuckDBFramework(DuckDBFramework):
    """Stands in for Spark/Iceberg: capable of running in a spawned MULTIPROCESSING worker."""

    @classmethod
    def supported_parallelization_modes(cls) -> set[ParallelizationMode]:
        return {ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING}

    def select_data_by_column_names(
        self,
        data: Any,
        selected_feature_names: Any,
        column_ordering: str | None = None,
        request_feature_order: list[str] | None = None,
    ) -> Any:
        """Stay lazy: return the DuckdbRelation as-is instead of DuckDBFramework's default
        pa.Table materialization, so a direct request of a DuckDB-native feature keeps its type."""
        return data


class _MpTfsRawValSource(FeatureGroup):
    """Source: emits mp_tfs_raw_val=[1,2,3] on PythonDictFramework, not PyArrowTable, to avoid
    aliasing with the PyArrowTable sink two hops downstream (mloda reuses one CFW instance per
    framework class per plan)."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"mp_tfs_raw_val"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mp_tfs_raw_val": [1, 2, 3]}


class _MpTfsDoubledDuckDBFG(FeatureGroup):
    """Doubles mp_tfs_raw_val via a DuckDB SQL projection that requires the connection.
    Plain FeatureGroup, not MatchData: MatchData embeds the live connection object into
    Options, which can't be pickled for a MULTIPROCESSING worker."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mp_tfs_raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_MultiprocessingConnectionDuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.project("*, mp_tfs_raw_val * 2 AS mp_tfs_doubled")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mp_tfs_doubled"}


class _MpTfsDoubledBackToPyArrowFG(FeatureGroup):
    """Re-emits mp_tfs_doubled on PyArrowTable as mp_tfs_final; the direct route needs a connection the parent must open itself from the registered spec."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mp_tfs_doubled")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("mp_tfs_final", data["mp_tfs_doubled"])

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mp_tfs_final"}


def _plugin_collector() -> PluginCollector:
    return PluginCollector.enabled_feature_groups(
        {_MpTfsRawValSource, _MpTfsDoubledDuckDBFG, _MpTfsDoubledBackToPyArrowFG}
    )


def _compute_frameworks() -> set[type[ComputeFramework]]:
    return {PythonDictFramework, _MultiprocessingConnectionDuckDBFramework, PyArrowTable}


@pytest.mark.timeout(30)
@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow not installed.")
def test_spec_connection_reaches_multiprocessing_worker(flight_server: Any) -> None:
    dac = DataAccessCollection(connections={ConnectionSpec(_MultiprocessingConnectionDuckDBFramework)})

    result = mloda.run_all(
        [Feature("mp_tfs_final")],
        compute_frameworks=_compute_frameworks(),
        plugin_collector=_plugin_collector(),
        data_access_collection=dac,
        parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        flight_server=flight_server,
    )

    assert result is not None
    assert len(result) == 1
    final = result[0]
    assert isinstance(final, pa.Table)
    assert "mp_tfs_final" in final.column_names
    assert final.column("mp_tfs_final").to_pylist() == [2, 4, 6]


@pytest.mark.timeout(30)
@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow not installed.")
def test_direct_request_of_connection_destination_materializes_in_parent(flight_server: Any) -> None:
    """No PyArrow re-emission hop: the parent must open its own connection from the spec to convert the flight-downloaded pa.Table back."""
    dac = DataAccessCollection(connections={ConnectionSpec(_MultiprocessingConnectionDuckDBFramework)})

    result = mloda.run_all(
        [Feature("mp_tfs_doubled")],
        compute_frameworks=_compute_frameworks(),
        plugin_collector=_plugin_collector(),
        data_access_collection=dac,
        parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        flight_server=flight_server,
    )

    assert result is not None
    assert len(result) == 1
    doubled = result[0]
    assert isinstance(doubled, DuckdbRelation)
    arrow = doubled.to_arrow_table()
    assert arrow.column("mp_tfs_doubled").to_pylist() == [2, 4, 6]


@pytest.mark.timeout(30)
@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow not installed.")
def test_live_connection_is_rejected_before_workers_start(flight_server: Any) -> None:
    setup_connection = duckdb.connect()
    dac = DataAccessCollection(connections={setup_connection})

    try:
        with pytest.raises(ValueError, match=r"ConnectionSpec\("):
            mloda.run_all(
                [Feature("mp_tfs_final")],
                compute_frameworks=_compute_frameworks(),
                plugin_collector=_plugin_collector(),
                data_access_collection=dac,
                parallelization_modes={ParallelizationMode.MULTIPROCESSING},
                flight_server=flight_server,
            )
    finally:
        setup_connection.close()
