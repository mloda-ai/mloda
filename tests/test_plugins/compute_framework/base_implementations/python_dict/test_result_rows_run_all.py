"""End-to-end test: ``result_rows`` unwraps a PythonDict ``mloda.run_all`` result (issue 717)."""

from typing import Any, Optional

from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import result_rows


class ResultRowsRootFeatureGroup(FeatureGroup):
    """Root PythonDict FeatureGroup emitting a two-row columnar dict for ``result_rows``."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"result_rows_e2e_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"result_rows_e2e_col": [1, 2]}


_ENABLED_RESULT_ROWS = PluginCollector.enabled_feature_groups({ResultRowsRootFeatureGroup})


def test_result_rows_unwraps_run_all_result(flight_server: Any) -> None:
    """``result_rows`` flattens the ``run_all`` partition list into the expected row dicts."""
    result = mloda.run_all(
        [Feature(name="result_rows_e2e_col")],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=_ENABLED_RESULT_ROWS,
        parallelization_modes={ParallelizationMode.SYNC},
        flight_server=flight_server,
    )

    assert result_rows(result) == [{"result_rows_e2e_col": 1}, {"result_rows_e2e_col": 2}]
