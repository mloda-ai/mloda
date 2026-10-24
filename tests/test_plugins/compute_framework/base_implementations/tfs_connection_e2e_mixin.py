"""End-to-end TFS connection-propagation mixin shared by all SQL frameworks.

The fix for issue #440 binds a DAC-resolved connection to the destination CFW
of a TransformFrameworkStep at setup time. This mixin exercises that wiring
through `mloda.run_all` so each SQL framework gets the same coverage as the
original DuckDB reproducer."""

import functools
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import pytest

from mloda.user import mloda, Feature, FeatureName, Options, DataAccessCollection, ParallelizationMode, PluginCollector
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.provider import DataCreatorFeatureGroup
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class TfsRawValPyArrowSource(DataCreatorFeatureGroup):
    """Shared source: emits raw_val=[1,2,3] on PyArrowTable so every SQL backend
    test exercises the same PyArrow -> <SQL framework> TFS edge."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"raw_val"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"raw_val": [1, 2, 3]}


_ROOT_SEEN: dict[str, Any] = {}


@functools.cache
def root_binding_groups(
    framework: type[ComputeFramework], double: Callable[[Any, str, str], Any]
) -> tuple[type[FeatureGroup], type[FeatureGroup], str]:
    """Build a root and a sink pinned to ``framework``; the sink records the connection of its input.

    Returns (root, sink, sink_feature_name). The root returns a dict that run_calculation must
    convert on the connection bound to the framework.
    """
    src = f"tfs_root_val_{framework.__name__}"
    dst = f"tfs_root_out_{framework.__name__}"

    class RootBindingRoot(DataCreatorFeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {src}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
            return {framework}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {src: [1, 2, 3]}

    class RootBindingSink(FeatureGroup):
        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature(src, options=options)}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
            return {framework}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            _ROOT_SEEN[dst] = data.connection
            return double(data, src, dst)

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {dst}

    return RootBindingRoot, RootBindingSink, dst


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

    @staticmethod
    def double_relation(data: Any, src: str, dst: str) -> Any:
        """Return ``data`` with column ``dst`` = ``src`` * 2, on the data's own connection."""
        raise NotImplementedError

    def test_a_root_runs_on_the_one_dac_connection_without_a_class_name_option(self, live_connection: Any) -> None:
        root, sink, dst = root_binding_groups(self.destination_framework_class, self.double_relation)
        _ROOT_SEEN.clear()
        result = mloda.run_all(
            [Feature(dst)],
            compute_frameworks=[self.destination_framework_class],
            plugin_collector=PluginCollector.enabled_feature_groups({root, sink}),
            data_access_collection=DataAccessCollection({live_connection}),
            parallelization_modes={ParallelizationMode.SYNC},
        )
        assert sorted(result[0].column(dst).to_pylist()) == [2, 4, 6]
        assert _ROOT_SEEN[dst] is live_connection

    def test_tfs_connection_reaches_destination_framework(self, live_connection: Any) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({self.source_fg_class, self.destination_fg_class})
        dac = DataAccessCollection(connections={live_connection})
        result = mloda.run_all(
            [Feature(self.destination_feature_name)],
            compute_frameworks=[self.source_framework_class, self.destination_framework_class],
            plugin_collector=plugin_collector,
            data_access_collection=dac,
            parallelization_modes={ParallelizationMode.SYNC},
        )
        assert result is not None
        assert len(result) == 1
        assert self.extract_doubled(result[0]) == [2, 4, 6]

    def test_a_connection_on_the_feature_needs_no_data_access_collection(self, live_connection: Any) -> None:
        root, sink, dst = root_binding_groups(self.destination_framework_class, self.double_relation)
        _ROOT_SEEN.clear()
        result = mloda.run_all(
            [Feature(dst, connection=live_connection)],
            compute_frameworks=[self.destination_framework_class],
            plugin_collector=PluginCollector.enabled_feature_groups({root, sink}),
            parallelization_modes={ParallelizationMode.SYNC},
        )
        assert sorted(result[0].column(dst).to_pylist()) == [2, 4, 6]
        assert _ROOT_SEEN[dst] is live_connection
