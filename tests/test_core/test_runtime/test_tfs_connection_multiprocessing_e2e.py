"""E2E: a connection-requiring destination framework opts into MULTIPROCESSING and
self-constructs its connection worker-side, without needing a real JVM or catalog
in the worker."""

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
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class _MultiprocessingConnectionDuckDBFramework(DuckDBFramework):
    """Opts into MULTIPROCESSING and self-constructs its connection worker-side instead
    of receiving the parent process's live object."""

    @classmethod
    def supported_parallelization_modes(cls) -> set[ParallelizationMode]:
        return {ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING}

    def set_framework_connection_object(self, framework_connection_object: Any | None = None) -> None:
        if framework_connection_object is None and self.framework_connection_object is None:
            framework_connection_object = duckdb.connect()
        super().set_framework_connection_object(framework_connection_object)


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
    """Re-emits mp_tfs_doubled on PyArrowTable as mp_tfs_final. Requesting mp_tfs_doubled
    directly would hit a separate, unrelated gap: materializing it in the parent process also
    needs a connection that the parent never binds. Routing through PyArrowTable sidesteps
    that without masking the worker-side bind this test exists to prove."""

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


@pytest.mark.timeout(30)
@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow not installed.")
def test_tfs_connection_reaches_multiprocessing_worker(flight_server: Any) -> None:
    setup_connection = duckdb.connect()
    plugin_collector = PluginCollector.enabled_feature_groups(
        {_MpTfsRawValSource, _MpTfsDoubledDuckDBFG, _MpTfsDoubledBackToPyArrowFG}
    )
    dac = DataAccessCollection(connections={setup_connection})

    result = mloda.run_all(
        [Feature("mp_tfs_final")],
        compute_frameworks={PythonDictFramework, _MultiprocessingConnectionDuckDBFramework, PyArrowTable},
        plugin_collector=plugin_collector,
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

    setup_connection.close()
