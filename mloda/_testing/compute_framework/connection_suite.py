"""Conformance suites for connection-bound ComputeFrameworks: pick_connection_from_dac and TFS propagation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import pytest

from mloda._testing.compute_framework.host import ComputeFrameworkSuite
from mloda._testing.compute_framework_shape import serves_connection
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.source import reclaim
from mloda.core.optional_dependency import require
from mloda.provider import ComputeFramework, DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import (
    DataAccessCollection,
    Feature,
    FeatureName,
    Options,
    ParallelizationMode,
    PluginCollector,
    mloda,
)


def root_binding_groups(
    framework: type[ComputeFramework], double: Callable[[Any, str, str], Any]
) -> tuple[type[FeatureGroup], type[FeatureGroup], str]:
    """Build a root and a sink pinned to ``framework``; returns (root, sink, sink_feature_name), reclaim-tracked."""
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
            return double(data, src, dst)

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {dst}

    return reclaim.track(RootBindingRoot), reclaim.track(RootBindingSink), dst


def _raw_val_source(source_framework: type[ComputeFramework]) -> type[DataCreatorFeatureGroup]:
    """A source emitting raw_val=[1,2,3] on ``source_framework``, tracked so it never outlives its test."""

    class TfsRawValSource(DataCreatorFeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"raw_val"}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
            return {source_framework}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {"raw_val": [1, 2, 3]}

    return reclaim.track(TfsRawValSource)


class ConnectionPickSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of pick_connection_from_dac and connection adoption on a connection-bound framework."""

    aspect = "connection"
    missing_connection_prefix: ClassVar[str] = "connection object is required"

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return serves_connection(framework)

    @pytest.fixture
    def second_adapter(self, adapter: FrameworkAdapter) -> Iterator[FrameworkAdapter]:
        with self.adapter_class() as other:
            yield other

    def second_connection(self, second_adapter: FrameworkAdapter) -> Any:
        """A second, distinct connection of the same framework type."""
        return second_adapter.connection

    def wrong_type_connection(self) -> Any:
        return object()

    def data_on_connection(self, adapter: FrameworkAdapter) -> Any:
        """A one-column frame living on ``adapter``'s connection."""
        pa = require("pyarrow", "building Arrow test tables")
        return adapter.from_arrow(pa.table({"a": [1]}))

    def test_returns_none_with_none(self, framework: Any) -> None:
        assert framework.pick_connection_from_dac(None) is None

    def test_returns_none_with_empty_collection(self, framework: Any) -> None:
        dac = DataAccessCollection(connections={})
        assert framework.pick_connection_from_dac(dac) is None

    def test_returns_none_with_wrong_type(self, framework: Any) -> None:
        dac = DataAccessCollection(connections={self.wrong_type_connection()})
        assert framework.pick_connection_from_dac(dac) is None

    def test_returns_matching_connection(self, framework: Any, adapter: FrameworkAdapter) -> None:
        dac = DataAccessCollection(connections={adapter.connection})
        assert framework.pick_connection_from_dac(dac) is adapter.connection

    def test_classmethod_is_pure(self, framework: Any, adapter: FrameworkAdapter) -> None:
        dac = DataAccessCollection(connections={adapter.connection})
        first = framework.pick_connection_from_dac(dac)
        second = framework.pick_connection_from_dac(dac)
        assert first is adapter.connection
        assert second is adapter.connection

    def test_raises_on_multiple_matches(
        self, framework: Any, adapter: FrameworkAdapter, second_adapter: FrameworkAdapter
    ) -> None:
        second = self.second_connection(second_adapter)
        assert adapter.connection is not second
        dac = DataAccessCollection(connections={adapter.connection, second})
        with pytest.raises(ValueError, match="Ambiguous resolve") as exc:
            framework.pick_connection_from_dac(dac)
        assert f"{framework.get_class_name()}: Ambiguous resolve" in str(exc.value)

    def test_hint_resolves_ambiguity(
        self, framework: Any, adapter: FrameworkAdapter, second_adapter: FrameworkAdapter
    ) -> None:
        """A ``data_access_handle`` hint on Options disambiguates between several matching connections."""
        second = self.second_connection(second_adapter)
        assert adapter.connection is not second
        dac = DataAccessCollection(connections={"primary": adapter.connection, "secondary": second})
        options = Options(context={"data_access_handle": "secondary"})
        assert framework.pick_connection_from_dac(dac, options=options) is second

    def test_connection_handle_picks_the_named_connection(
        self, framework: Any, adapter: FrameworkAdapter, second_adapter: FrameworkAdapter
    ) -> None:
        second = self.second_connection(second_adapter)
        dac = DataAccessCollection(connections={"a": adapter.connection, "b": second})
        options = Options(group={"connection_handle": "b"})
        assert framework.pick_connection_from_dac(dac, options=options) is second

    def test_adopt_connection_rejects_data_on_another_connection_than_the_bound_one(
        self, framework: Any, adapter: FrameworkAdapter, second_adapter: FrameworkAdapter
    ) -> None:
        fw = framework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        fw.set_framework_connection_object(adapter.connection)
        foreign = self.data_on_connection(second_adapter)
        with pytest.raises(ValueError, match=f"{framework.__name__}.*connection_handle"):
            fw._adopt_connection(foreign)

    def test_adopt_connection_accepts_the_bound_connection_and_adopts_when_unbound(
        self, framework: Any, adapter: FrameworkAdapter
    ) -> None:
        fw = framework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        fw.fallback_connection_object = object()
        data = self.data_on_connection(adapter)
        fw._adopt_connection(data)
        assert fw.framework_connection_object is adapter.connection
        fw._adopt_connection(data)

    def test_missing_connection_error_names_the_dac_and_connection_handle(self, framework: Any) -> None:
        fw = framework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        with pytest.raises(
            ValueError, match=f"{self.missing_connection_prefix}.*DataAccessCollection.*connection_handle"
        ):
            fw.set_framework_connection_object(None)


class ConnectionEndToEndSuite(ComputeFrameworkSuite, suite_base=True):
    """Verifies the DAC connection threads to a TFS destination framework end-to-end through run_all."""

    aspect = "connection"
    required_hooks = ("destination_fg_class", "extract_doubled")
    destination_feature_name: ClassVar[str] = "tfs_doubled"

    destination_fg_class: type[FeatureGroup]

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return serves_connection(framework)

    @pytest.fixture(autouse=True)
    # The "_a_" prefix sorts it first among autouse fixtures, so its mark is the outermost.
    def _a_reclaim_suite_built_classes(self) -> Iterator[None]:
        yield from reclaim.reclaimed_scope()

    def destination_framework_class(self, framework: type[ComputeFramework]) -> type[ComputeFramework]:
        """The framework the destination feature groups run on; defaults to the host's framework."""
        return framework

    def source_framework_class(self) -> type[ComputeFramework]:
        """The framework the raw_val source runs on; defaults to PyArrowTable."""
        from mloda.user.pyarrow import PyArrowTable

        return PyArrowTable

    def extract_doubled(self, final: Any) -> list[int]:
        raise NotImplementedError

    @staticmethod
    def double_relation(data: Any, src: str, dst: str) -> Any:
        """Return ``data`` with column ``dst`` = ``src`` * 2, on the data's own connection."""
        raise NotImplementedError

    def _root_groups(
        self, destination: type[ComputeFramework], seen: dict[str, Any]
    ) -> tuple[type[FeatureGroup], type[FeatureGroup], str]:
        dst = f"tfs_root_out_{destination.__name__}"

        def record(data: Any, src: str, out: str) -> Any:
            seen[dst] = data.connection
            return self.double_relation(data, src, out)

        return root_binding_groups(destination, record)

    def test_a_root_runs_on_the_one_dac_connection_without_a_class_name_option(
        self, adapter: FrameworkAdapter, framework: type[ComputeFramework]
    ) -> None:
        destination = self.destination_framework_class(framework)
        seen: dict[str, Any] = {}
        root, sink, dst = self._root_groups(destination, seen)
        result = mloda.run_all(
            [Feature(dst)],
            compute_frameworks=[destination],
            plugin_collector=PluginCollector.enabled_feature_groups({root, sink}),
            data_access_collection=DataAccessCollection({adapter.connection}),
            parallelization_modes={ParallelizationMode.SYNC},
        )
        assert sorted(row[dst] for row in self._result_rows(adapter, result[0])) == [2, 4, 6]
        assert seen[dst] is adapter.connection

    def test_tfs_connection_reaches_destination_framework(
        self, adapter: FrameworkAdapter, framework: type[ComputeFramework]
    ) -> None:
        destination = self.destination_framework_class(framework)
        source_framework = self.source_framework_class()
        source = _raw_val_source(source_framework)
        result = mloda.run_all(
            [Feature(self.destination_feature_name)],
            compute_frameworks=[source_framework, destination],
            plugin_collector=PluginCollector.enabled_feature_groups({source, self.destination_fg_class}),
            data_access_collection=DataAccessCollection(connections={adapter.connection}),
            parallelization_modes={ParallelizationMode.SYNC},
        )
        assert result is not None
        assert len(result) == 1
        assert self.extract_doubled(result[0]) == [2, 4, 6]

    def test_a_connection_on_the_feature_needs_no_data_access_collection(
        self, adapter: FrameworkAdapter, framework: type[ComputeFramework]
    ) -> None:
        destination = self.destination_framework_class(framework)
        seen: dict[str, Any] = {}
        root, sink, dst = self._root_groups(destination, seen)
        result = mloda.run_all(
            [Feature(dst, connection=adapter.connection)],
            compute_frameworks=[destination],
            plugin_collector=PluginCollector.enabled_feature_groups({root, sink}),
            parallelization_modes={ParallelizationMode.SYNC},
        )
        assert sorted(row[dst] for row in self._result_rows(adapter, result[0])) == [2, 4, 6]
        assert seen[dst] is adapter.connection
