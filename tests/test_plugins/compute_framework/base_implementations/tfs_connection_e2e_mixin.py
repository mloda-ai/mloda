"""End-to-end TFS connection-propagation mixin shared by all SQL frameworks.

The fix for issue #440 binds a DAC-resolved connection to the destination CFW
of a TransformFrameworkStep at setup time. This mixin exercises that wiring
through `mloda.run_all` so each SQL framework gets the same coverage as the
original DuckDB reproducer."""

from abc import abstractmethod
from typing import Any, Optional

import pytest

from mloda.user import mloda, Feature, DataAccessCollection, ParallelizationMode, PluginCollector
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.provider import BaseInputData
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class TfsRawValPyArrowSource(FeatureGroup):
    """Shared source: emits raw_val=[1,2,3] on PyArrowTable so every SQL backend
    test exercises the same PyArrow -> <SQL framework> TFS edge."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"raw_val"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"raw_val": [1, 2, 3]}


class TfsConnectionEndToEndMixin:
    """Verifies the DAC connection threads to a TFS destination CFW end-to-end.

    Subclass contract:
      - `destination_framework_class` attribute (the SQL ComputeFramework subclass)
      - `destination_fg_class` attribute (a FeatureGroup whose calculate_feature
        REQUIRES framework_connection_object; without the fix the test fails)
      - `live_connection` fixture yielding a real connection of the destination
        framework's native type
      - `extract_doubled(final)` returns the [2, 4, 6] list from the result
    """

    destination_framework_class: type[ComputeFramework]
    destination_fg_class: type[FeatureGroup]
    source_framework_class: type[ComputeFramework] = PyArrowTable
    source_fg_class: type[FeatureGroup] = TfsRawValPyArrowSource
    destination_feature_name: str = "tfs_doubled"

    @pytest.fixture
    @abstractmethod
    def live_connection(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def extract_doubled(self, final: Any) -> list[int]:
        raise NotImplementedError

    def test_tfs_connection_reaches_destination_framework(self, live_connection: Any) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({self.source_fg_class, self.destination_fg_class})
        dac = DataAccessCollection(connections={live_connection})
        result = mloda.run_all(
            [Feature(self.destination_feature_name)],
            compute_frameworks={self.source_framework_class, self.destination_framework_class},
            plugin_collector=plugin_collector,
            data_access_collection=dac,
            parallelization_modes={ParallelizationMode.SYNC},
        )
        assert result is not None
        assert len(result) == 1
        assert self.extract_doubled(result[0]) == [2, 4, 6]
