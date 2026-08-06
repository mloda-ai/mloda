"""GlobalFilter must own the filter Feature it stores, not the queue object the planner rewrites."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.filter.single_filter import SingleFilter
from mloda.provider import DataCreator
from mloda.user import FeatureName, FilterType, GlobalFilter, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


# The gfo_ prefix keeps these names unique repo-wide.
GFO_HOST = "gfo_owned_host"
GFO_FILTER_ONLY = "gfo_owned_filter_only"


class GfoOwnedRoot(FeatureGroup):
    """Root serving the requested host column plus a column only a filter ever reaches."""

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({GFO_HOST, GFO_FILTER_ONLY})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {GFO_HOST: [1], GFO_FILTER_ONLY: [1]}


_ENABLED = PluginCollector.enabled_feature_groups({GfoOwnedRoot})
_COLLECTION_KEY = (GfoOwnedRoot, FeatureName(GFO_HOST))


def _prepared() -> tuple[set[Feature], GlobalFilter]:
    """Plan one request whose only filter targets a column that is never requested."""
    global_filter = GlobalFilter()
    global_filter.add_filter(GFO_FILTER_ONLY, FilterType.EQUAL, {"value": 1})

    session = mloda.prepare(
        [GFO_HOST],
        compute_frameworks={PythonDictFramework},
        plugin_collector=_ENABLED,
        global_filter=global_filter,
    )

    engine = session.engine
    assert engine is not None, "prepare must build an engine"
    assert engine.global_filter is global_filter, "the session must plan against the caller's GlobalFilter"
    return engine.feature_group_collection[GfoOwnedRoot], global_filter


def _stored_filter(global_filter: GlobalFilter) -> SingleFilter:
    stored = global_filter.collection[_COLLECTION_KEY]
    assert len(stored) == 1, f"exactly one filter must attach to the host: {stored!r}"
    return next(iter(stored))


def _probe_key(global_filter: GlobalFilter) -> tuple[type[FeatureGroup], FeatureName, UUID]:
    """The probe key for the host feature: its uuid only exists after planning."""
    keys = [key for key in global_filter.probes if key[0] is GfoOwnedRoot and key[1] == GFO_HOST]
    assert len(keys) == 1, f"exactly one probe must be recorded for the host: {list(global_filter.probes)!r}"
    return keys[0]


def test_the_stored_filter_feature_is_not_the_queue_feature() -> None:
    queue, global_filter = _prepared()
    stored_feature = _stored_filter(global_filter).filter_feature

    assert not any(feature is stored_feature for feature in queue), (
        "the stored filter feature must not be the queue Feature the planner rewrites"
    )
    assert any(feature == stored_feature for feature in queue), (
        "the stored filter feature must stay value-equal to its queue twin"
    )


def test_a_framework_rewrite_of_the_queue_twin_does_not_strand_the_stored_filter() -> None:
    """The stored filter stays findable after the rewrite, without any rehash step."""
    queue, global_filter = _prepared()
    stored = _stored_filter(global_filter)

    twins = [feature for feature in queue if feature == stored.filter_feature]
    assert len(twins) == 1, f"exactly one queue Feature must equal the stored filter feature: {twins!r}"
    # What ResolveComputeFrameworks.links does to every queue Feature it resolves.
    twins[0].compute_frameworks = {PyArrowTable}

    assert stored in global_filter.collection[_COLLECTION_KEY], (
        "the rewrite must not lose the stored filter from its own collection set"
    )
    assert stored in global_filter.probes[_probe_key(global_filter)], (
        "the rewrite must not lose the stored filter from its own probes set"
    )


def test_the_filter_still_attaches_to_the_host_feature() -> None:
    """Guard: without a matched filter the other tests would pass vacuously."""
    _, global_filter = _prepared()

    collected = {single_filter.name for single_filter in global_filter.collection[_COLLECTION_KEY]}
    probed = {single_filter.name for single_filter in global_filter.probes[_probe_key(global_filter)]}

    assert collected == {GFO_FILTER_ONLY}, f"the collection must name the resolved filter feature: {collected!r}"
    assert probed == {GFO_FILTER_ONLY}, f"the probes must name the resolved filter feature: {probed!r}"
