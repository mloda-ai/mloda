"""Every SingleFilter GlobalFilter stores must stay findable in its own hash-keyed set after planning.

A rename shifts a matched filter's hash before record_probe merges the caller's set, and an option
value that is itself a Feature gets re-stamped by a second host's input resolution after storage.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.filter.single_filter import SingleFilter
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, DefaultOptionKeys, FeatureGroup, FeatureSet
from mloda.user import (
    Feature,
    FeatureName,
    Features,
    FilterType,
    GlobalFilter,
    Options,
    ParallelizationMode,
    PluginCollector,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_tooling import MlodaTestRunner

# The sphr_ prefix (rename vector) and sphn_ prefix (nested option vector) keep these names unique repo-wide.
SPHR_VALUE = "sphr_value"
SPHR_EVENT_TIME = "sphr_event_time"
SPHR_EVENT_TIME_UTC = "sphr_event_time_utc"

SPHN_LEAF = "sphn_leaf"
SPHN_HOST_ONE = "sphn_host_one"
SPHN_HOST_TWO = "sphn_host_two"
SPHN_FILTER = "sphn_filter_col"
SPHN_VARIANT = "sphn_variant"
SPHN_CUSTOM_SOURCE = "sphn_custom_source"


def _stranded(stored_sets: Iterable[set[SingleFilter]]) -> list[str]:
    """Names of the stored filters their own set can no longer find."""
    return [single.name for stored in stored_sets for single in stored if single not in stored]


def _stored_filters(global_filter: GlobalFilter) -> list[SingleFilter]:
    """Every stored filter, from both hash-keyed stores."""
    return [
        single
        for stored in chain(global_filter.collection.values(), global_filter.probes.values())
        for single in stored
    ]


class SphrRenamingRoot(FeatureGroup):
    """Root that renames the filter column via set_feature_name."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({SPHR_VALUE, SPHR_EVENT_TIME})

    def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
        if str(feature_name) == SPHR_EVENT_TIME:
            return FeatureName(SPHR_EVENT_TIME_UTC)
        return feature_name

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {SPHR_VALUE: [1, 2, 3], SPHR_EVENT_TIME_UTC: [10, 20, 30]}


_SPHR_ENABLED = PluginCollector.enabled_feature_groups({SphrRenamingRoot})


def test_a_renaming_run_keeps_every_probed_filter_findable() -> None:
    """The rename lands while the matched set is already built, so record_probe merges a stale hash."""
    global_filter = GlobalFilter()
    global_filter.add_filter(SPHR_EVENT_TIME, FilterType.MIN, {"value": 15})

    MlodaTestRunner.run_api(
        Features([Feature(SPHR_VALUE)]),
        compute_frameworks={PythonDictFramework},
        parallelization_modes={ParallelizationMode.SYNC},
        global_filter=global_filter,
        plugin_collector=_SPHR_ENABLED,
    )

    probed = {single.name for stored in global_filter.probes.values() for single in stored}
    assert probed == {SPHR_EVENT_TIME_UTC}, f"the rename must reach the recorded filter: {probed!r}"
    assert _stranded(global_filter.probes.values()) == [], "a probed filter must be findable in its own set"
    assert _stranded(global_filter.collection.values()) == [], "a collected filter must be findable in its own set"


class SphnRoot(FeatureGroup):
    """Root serving the leaf column the shared nested Feature names."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({SPHN_LEAF})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {SPHN_LEAF: [1, 2, 3]}


class SphnConsumer(FeatureGroup):
    """Consumer taking its input features from the in_features key or from an ordinary custom key."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) in {SPHN_HOST_ONE, SPHN_HOST_TWO, SPHN_FILTER}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        for key in (DefaultOptionKeys.in_features.value, SPHN_CUSTOM_SOURCE):
            value = options.get(key)
            if isinstance(value, Feature):
                return {value}
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {str(feature.name): list(data[SPHN_LEAF]) for feature in features.features}


_SPHN_ENABLED = PluginCollector.enabled_feature_groups({SphnRoot, SphnConsumer})


def _plan_two_hosts_sharing(key: str, shared: Feature, copy_features: bool = True) -> GlobalFilter:
    """Plan two hosts whose ``key`` option value is the one shared Feature, under one global filter."""
    global_filter = GlobalFilter()
    global_filter.add_filter(SPHN_FILTER, FilterType.MIN, {"value": 1})

    mloda.prepare(
        [
            Feature(SPHN_HOST_ONE, Options(group={key: shared, SPHN_VARIANT: 1})),
            Feature(SPHN_HOST_TWO, Options(group={key: shared, SPHN_VARIANT: 2})),
        ],
        compute_frameworks={PythonDictFramework},
        plugin_collector=_SPHN_ENABLED,
        global_filter=global_filter,
        copy_features=copy_features,
    )
    return global_filter


def _not_carrying(global_filter: GlobalFilter, key: str) -> list[str]:
    """Stored filters that never received the host's ``key``, whichever category it belongs to."""
    return [single.name for single in _stored_filters(global_filter) if key not in single.filter_feature.options]


def _aliasing(global_filter: GlobalFilter, shared: Feature) -> list[str]:
    """Stored filters holding the host's shared Feature itself, not a copy."""
    return [
        single.name
        for single in _stored_filters(global_filter)
        if any(value is shared for _key, value in single.filter_feature.options.items())
    ]


def _assert_two_hosts_and_no_strand(global_filter: GlobalFilter, key: str) -> None:
    """The invariant every key and copy_features variant shares."""
    hosts = {str(name) for _group, name in global_filter.collection}
    assert hosts == {SPHN_HOST_ONE, SPHN_HOST_TWO}, f"the filter must attach to both hosts: {hosts!r}"

    assert _not_carrying(global_filter, key) == [], f"every stored filter must carry the host's '{key}'"
    assert _stranded(global_filter.collection.values()) == [], "a collected filter must be findable in its own set"
    assert _stranded(global_filter.probes.values()) == [], "a probed filter must be findable in its own set"


# Any option key can hold a shared Feature, so no repair may be scoped to the input-feature key.
SPHN_SHARING_KEYS = [DefaultOptionKeys.in_features.value, SPHN_CUSTOM_SOURCE]


@pytest.mark.parametrize("key", SPHN_SHARING_KEYS)
def test_a_shared_input_feature_strands_no_stored_filter_on_the_default_path(key: str) -> None:
    """The normal run copies the caller's features, and the strand reproduces there just the same."""
    global_filter = _plan_two_hosts_sharing(key, Feature(SPHN_LEAF, forward_group=False))

    _assert_two_hosts_and_no_strand(global_filter, key)


@pytest.mark.parametrize("key", SPHN_SHARING_KEYS)
def test_a_stored_filter_holds_a_copy_of_the_hosts_shared_input_feature(key: str) -> None:
    """copy_features=False so ``shared`` IS the planned object: a deepcopied twin would pass the identity check."""
    shared = Feature(SPHN_LEAF, forward_group=False)

    global_filter = _plan_two_hosts_sharing(key, shared, copy_features=False)

    _assert_two_hosts_and_no_strand(global_filter, key)
    assert _aliasing(global_filter, shared) == [], "a stored filter must hold a copy, not the host's shared Feature"
