"""Pin ``SingleFilter`` owning its filter feature instead of aliasing the caller's object (#910).

``handle_filter_feature`` stores the caller's ``Feature`` as-is, so anything that later rebinds or
mutates one of the containers that feature hashes over shifts the hash of a ``SingleFilter`` already
sitting in the public ``GlobalFilter.filters`` set. Three public paths reach it with
``copy_features=False``: intake materializes a declared group default onto the shared feature,
``strict_type_enforcement=True`` adds a group key to the shared ``Options`` in place, and that same
in-place write lands on the ``child_options`` the engine stamped onto a cached input feature. Either
way the set loses its own member and a value-equal ``add_filter`` stops deduplicating.

``compute_frameworks`` is hashed too, and a caller can write it in place at any time, so the filter
feature owns that set as well (#924).
"""

from __future__ import annotations

from typing import Any, Optional

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator, PropertySpec
from mloda.user import DataType, Feature, FilterType, GlobalFilter, Options, PluginCollector, SingleFilter, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework


# The sfo_ prefix keeps every key and feature name unique to this module.
SFO_GRP_KEY = "sfo_grp_key"
SFO_GRP_DEFAULT = "sfo_grp_default_val"
SFO_GRP_FEATURE = "sfo_grp_shared"
SFO_TYPED_FEATURE = "sfo_typed_shared"
SFO_UNIT_FEATURE = "sfo_unit_feature"
SFO_UNIT_KEY = "sfo_unit_key"
SFO_LATE_KEY = "sfo_late_key"
SFO_CACHED_FEATURE = "sfo_cached_input"
SFO_CONSUMER_FEATURE = "sfo_cached_consumer"
SFO_CONSUMER_KEY = "sfo_consumer_key"
SFO_LEAF_FEATURE = "sfo_leaf_feature"
SFO_LEAF_KEY = "sfo_leaf_key"
SFO_CFW_FEATURE = "sfo_cfw_feature"


def _make_shared_fg(name: str, property_mapping: dict[str, PropertySpec]) -> type[FeatureGroup]:
    """A throwaway root feature group serving one name and reporting the filters it received."""

    class SfoSharedFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = property_mapping

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({name})

        @classmethod
        def final_filters(cls) -> bool:
            # The payload is not filterable data: report features.filters inline instead of running
            # post-calculation row elimination against it.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            delivered = len(features.filters) if features.filters else 0
            return {str(feature.name): [delivered] for feature in features.features}

    return SfoSharedFeatureGroup


def _observe(shared: Feature, twin: Feature, feature_group: type[FeatureGroup], strict: bool) -> dict[str, Any]:
    """Share one ``Feature`` between ``add_filter`` and an uncopied run; return a plain-data view.

    Every object referencing the throwaway feature group is dropped from this frame before returning,
    so a failing assert in the caller cannot pin it into a traceback and trip the no-leak fixture on
    top of the real failure.
    """
    column = str(shared.name)
    global_filter = GlobalFilter()
    global_filter.add_filter(shared, FilterType.EQUAL, {"value": 1})
    stored = next(iter(global_filter.filters))

    observed: dict[str, Any] = {
        "aliases_caller": stored.filter_feature is shared,
        "member_before": stored in global_filter.filters,
    }
    results = mloda.run_all(
        [shared],
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups({feature_group}),
        global_filter=global_filter,
        copy_features=False,
        strict_type_enforcement=strict,
    )
    frames = list(results)

    observed["member_after"] = stored in global_filter.filters
    observed["stored_group"] = dict(stored.filter_feature.options.group)
    observed["caller_group"] = dict(shared.options.group)
    observed["filters"] = len(global_filter.filters)
    # A declared twin of the original request: value-equal to what was added, so it must not grow the set.
    global_filter.add_filter(twin, FilterType.EQUAL, {"value": 1})
    observed["filters_after_readd"] = len(global_filter.filters)
    observed["collected"] = {str(name): len(entries) for (_group, name), entries in global_filter.collection.items()}
    observed["delivered"] = [frame[column][0] for frame in frames if column in frame]

    del shared, twin, feature_group, global_filter, stored, results, frames
    return observed


def _run_group_key_default() -> dict[str, Any]:
    """One feature declaring a group key as None, served by a group declaring a concrete default."""
    return _observe(
        Feature(SFO_GRP_FEATURE, Options(group={SFO_GRP_KEY: None})),
        Feature(SFO_GRP_FEATURE, Options(group={SFO_GRP_KEY: None})),
        _make_shared_fg(
            SFO_GRP_FEATURE,
            {SFO_GRP_KEY: PropertySpec("A group concrete default.", context=False, default=SFO_GRP_DEFAULT)},
        ),
        strict=False,
    )


def _run_strict_type_enforcement() -> dict[str, Any]:
    """One typed feature, no declared default: strict_type_enforcement adds the group key in place."""
    return _observe(
        Feature(SFO_TYPED_FEATURE, Options(), data_type=DataType.INT64),
        Feature(SFO_TYPED_FEATURE, Options(), data_type=DataType.INT64),
        _make_shared_fg(SFO_TYPED_FEATURE, {}),
        strict=True,
    )


def test_group_key_shared_filter_stays_findable_in_the_global_filter() -> None:
    """The shared filter stays findable in ``GlobalFilter.filters`` after the run (fails pre-fix:
    intake materializes the declared default onto the aliased feature, shifting the stored hash)."""
    observed = _run_group_key_default()
    assert observed["member_before"] is True, f"the filter must be in its own set before the run: {observed!r}"
    assert observed["member_after"] is True, f"the filter must stay findable in its own set: {observed!r}"


def test_group_key_shared_filter_keeps_its_declared_options() -> None:
    """The filter keeps the options it was declared with, while the caller's feature is still filled
    (fails pre-fix: both views are the same object, so the run rewrites the filter too)."""
    observed = _run_group_key_default()
    assert observed["stored_group"] == {SFO_GRP_KEY: None}, f"the filter must own its options: {observed!r}"
    assert observed["caller_group"] == {SFO_GRP_KEY: SFO_GRP_DEFAULT}, (
        f"intake must still materialize the caller's feature: {observed!r}"
    )


def test_group_key_shared_filter_dedupes_a_value_equal_readd_after_the_run() -> None:
    """Re-adding a value-equal filter after the run does not grow the set (fails pre-fix: the stored
    entry drifted to the materialized options, so the declared twin is no longer equal to it)."""
    observed = _run_group_key_default()
    assert observed["filters"] == 1, f"the run must not add filters: {observed!r}"
    assert observed["filters_after_readd"] == 1, f"a value-equal re-add must deduplicate: {observed!r}"


def test_group_key_shared_filter_is_still_matched_and_delivered() -> None:
    """Guard, passes pre-fix: the filter must still match its host and reach the feature group."""
    observed = _run_group_key_default()
    assert observed["collected"] == {SFO_GRP_FEATURE: 1}, f"the run must collect exactly one filter: {observed!r}"
    assert observed["delivered"] == [1], f"the feature group must receive exactly one filter: {observed!r}"


def test_strict_type_enforcement_shared_filter_stays_findable_in_the_global_filter() -> None:
    """A typed shared feature needs no declared default: strict_type_enforcement mutates the shared
    Options in place (fails pre-fix: the added group key shifts the stored filter's hash)."""
    observed = _run_strict_type_enforcement()
    assert observed["member_before"] is True, f"the filter must be in its own set before the run: {observed!r}"
    assert observed["member_after"] is True, f"the filter must stay findable in its own set: {observed!r}"


def test_strict_type_enforcement_shared_filter_dedupes_a_value_equal_readd_after_the_run() -> None:
    """Re-adding a value-equal typed filter after the run does not grow the set (fails pre-fix)."""
    observed = _run_strict_type_enforcement()
    assert observed["stored_group"] == {}, f"the filter must own its options: {observed!r}"
    assert observed["filters_after_readd"] == 1, f"a value-equal re-add must deduplicate: {observed!r}"


def test_strict_type_enforcement_shared_filter_is_still_matched_and_delivered() -> None:
    """Guard, passes pre-fix: the typed filter must still match its host and reach the feature group."""
    observed = _run_strict_type_enforcement()
    assert observed["collected"] == {SFO_TYPED_FEATURE: 1}, f"the run must collect exactly one filter: {observed!r}"
    assert observed["delivered"] == [1], f"the feature group must receive exactly one filter: {observed!r}"


def _make_cached_input_chain() -> tuple[Feature, type[FeatureGroup], type[FeatureGroup]]:
    """A consumer whose ``input_features`` hands out one cached ``Feature``, plus the root serving it."""
    cached = Feature(SFO_CACHED_FEATURE)

    class SfoCachedRoot(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({SFO_CACHED_FEATURE})

        @classmethod
        def final_filters(cls) -> bool:
            # The payload is not filterable data: report features.filters inline instead of running
            # post-calculation row elimination against it.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {SFO_CACHED_FEATURE: [len(features.filters) if features.filters else 0]}

    class SfoCachedConsumer(FeatureGroup):
        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
            # The same object every call, as a plugin holding a module-level or memoized Feature does.
            return {cached}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            return str(feature_name) == SFO_CONSUMER_FEATURE

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {SFO_CONSUMER_FEATURE: list(data[SFO_CACHED_FEATURE])}

    return cached, SfoCachedRoot, SfoCachedConsumer


def _run_cached_input_feature() -> dict[str, Any]:
    """Add a filter on the cached input feature between two runs sharing one requested ``Feature``.

    Run one stamps ``cached.child_options`` with the consumer's live ``Options``; run two adds
    ``strict_type_enforcement`` to that very object in place. Every object referencing the throwaway
    feature groups is dropped from this frame before returning, so a failing assert in the caller
    cannot pin one into a traceback and trip the no-leak fixture.
    """
    cached, root_fg, consumer_fg = _make_cached_input_chain()
    collector = PluginCollector.enabled_feature_groups({root_fg, consumer_fg})
    consumer = Feature(SFO_CONSUMER_FEATURE, Options(group={SFO_CONSUMER_KEY: "v"}), data_type=DataType.INT64)

    mloda.run_all(
        [consumer],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
        copy_features=False,
    )

    global_filter = GlobalFilter()
    global_filter.add_filter(cached, FilterType.EQUAL, {"value": 1})
    stored = next(iter(global_filter.filters))
    observed: dict[str, Any] = {
        "stamped": cached.child_options is consumer.options,
        "member_before": stored in global_filter.filters,
    }

    frames = list(
        mloda.run_all(
            [consumer],
            compute_frameworks={PythonDictFramework},
            plugin_collector=collector,
            global_filter=global_filter,
            copy_features=False,
            strict_type_enforcement=True,
        )
    )

    observed["member_after"] = stored in global_filter.filters
    stored_child = stored.filter_feature.child_options
    observed["stored_child_group"] = dict(stored_child.group) if stored_child is not None else None
    observed["delivered"] = [frame[SFO_CONSUMER_FEATURE][0] for frame in frames if SFO_CONSUMER_FEATURE in frame]

    del cached, root_fg, consumer_fg, collector, consumer, global_filter, stored, stored_child, frames
    return observed


def test_cached_input_feature_filter_stays_findable_in_the_global_filter() -> None:
    """The filter on a cached input feature stays findable after a later run (fails pre-fix: the
    stored feature shares the consumer's Options as child_options, so the in-place strict write
    shifts its hash)."""
    observed = _run_cached_input_feature()
    assert observed["stamped"] is True, f"the engine must stamp the consumer's live Options: {observed!r}"
    assert observed["member_before"] is True, f"the filter must be in its own set before the run: {observed!r}"
    assert observed["member_after"] is True, f"the filter must stay findable in its own set: {observed!r}"


def test_cached_input_feature_filter_keeps_the_child_options_it_was_stored_with() -> None:
    """The stored filter keeps the child_options snapshot taken at add_filter (fails pre-fix)."""
    observed = _run_cached_input_feature()
    assert observed["stored_child_group"] == {SFO_CONSUMER_KEY: "v"}, (
        f"the filter must own its child_options: {observed!r}"
    )


def test_cached_input_feature_filter_is_still_matched_and_delivered() -> None:
    """Guard, passes pre-fix: the filter must still reach the root feature group through the chain."""
    observed = _run_cached_input_feature()
    assert observed["delivered"] == [1], f"the root feature group must receive exactly one filter: {observed!r}"


def _unit_feature() -> Feature:
    """A plain feature with one declared group key, for the engine-free ownership pins."""
    return Feature(SFO_UNIT_FEATURE, Options(group={SFO_UNIT_KEY: "declared"}))


def test_single_filter_does_not_alias_the_caller_feature() -> None:
    """A SingleFilter owns its feature: value-equal to the caller's, never the same object."""
    feature = _unit_feature()
    single_filter = SingleFilter(feature, FilterType.EQUAL, {"value": 1})
    assert single_filter.filter_feature is not feature, "the filter must not alias the caller's Feature"
    assert single_filter.filter_feature == feature, "the filter's own feature must equal the caller's"


def test_single_filter_hash_survives_a_mutation_of_the_caller_feature() -> None:
    """Mutating the caller's Options after construction must not shift the filter's hash."""
    feature = _unit_feature()
    single_filter = SingleFilter(feature, FilterType.EQUAL, {"value": 1})
    before = hash(single_filter)
    feature.options.add_to_group(SFO_LATE_KEY, "late")
    assert hash(single_filter) == before, "a caller-side mutation must not shift the filter's hash"


def test_single_filter_stays_in_its_set_when_the_caller_feature_changes() -> None:
    """A stored filter stays findable after the caller mutates and then rebinds its Options."""
    feature = _unit_feature()
    single_filter = SingleFilter(feature, FilterType.EQUAL, {"value": 1})
    holder = {single_filter}
    feature.options.add_to_group(SFO_LATE_KEY, "late")
    assert single_filter in holder, "a caller-side mutation must not lose the filter from its set"
    feature.options = Options(group={SFO_UNIT_KEY: "rebound"})
    assert single_filter in holder, "a caller-side rebind must not lose the filter from its set"


def _cfw_feature() -> Feature:
    """A plain feature pinned to one compute framework, for the compute_frameworks ownership pins."""
    return Feature(SFO_CFW_FEATURE, compute_framework=PythonDictFramework.get_class_name())


def test_single_filter_does_not_alias_the_caller_features_compute_frameworks() -> None:
    """A SingleFilter owns that set too: value-equal to the caller's, never the same object."""
    feature = _cfw_feature()
    single_filter = SingleFilter(feature, FilterType.EQUAL, {"value": 1})
    assert single_filter.filter_feature.compute_frameworks == feature.compute_frameworks, (
        "the filter's own compute_frameworks must equal the caller's"
    )
    assert single_filter.filter_feature.compute_frameworks is not feature.compute_frameworks, (
        "the filter must not alias the caller's compute_frameworks set"
    )


def test_single_filter_hash_survives_an_in_place_compute_frameworks_write() -> None:
    """Adding a framework to the caller's set in place must not shift the filter's hash."""
    feature = _cfw_feature()
    single_filter = SingleFilter(feature, FilterType.EQUAL, {"value": 1})
    before = hash(single_filter)
    assert feature.compute_frameworks is not None
    feature.compute_frameworks.add(SqliteFramework)
    assert hash(single_filter) == before, "a caller-side in-place write must not shift the filter's hash"


def test_single_filter_stays_in_its_set_when_the_caller_adds_a_compute_framework() -> None:
    """A stored filter stays findable after the caller writes compute_frameworks in place, then rebinds."""
    feature = _cfw_feature()
    single_filter = SingleFilter(feature, FilterType.EQUAL, {"value": 1})
    holder = {single_filter}
    assert feature.compute_frameworks is not None
    feature.compute_frameworks.add(SqliteFramework)
    assert single_filter in holder, "a caller-side in-place write must not lose the filter from its set"
    feature.compute_frameworks = {SqliteFramework}
    assert single_filter in holder, "a caller-side rebind must not lose the filter from its set"


def test_string_filter_feature_still_builds_a_named_feature() -> None:
    """Guard, passes pre-fix: a str filter feature still becomes a Feature and still names the filter."""
    single_filter = SingleFilter(SFO_UNIT_FEATURE, FilterType.RANGE, {"min": 1, "max": 2})
    assert single_filter.name == SFO_UNIT_FEATURE
    assert str(single_filter.filter_feature.name) == SFO_UNIT_FEATURE


def test_independently_built_filters_stay_value_equal_and_dedupe() -> None:
    """Guard, passes pre-fix: equality stays by value, so equal filters still collapse in a set."""
    first = SingleFilter(_unit_feature(), FilterType.EQUAL, {"value": 1})
    second = SingleFilter(_unit_feature(), FilterType.EQUAL, {"value": 1})
    assert first == second, "filters with equal feature, type and parameter must compare equal"
    assert len({first, second}) == 1, "equal filters must deduplicate in a set"


class _SfoUnhashableLeaf:
    """An unhashable non-container option value; defining __eq__ alone drops __hash__ to None."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SfoUnhashableLeaf)


def _leaf_feature(leaf: _SfoUnhashableLeaf) -> Feature:
    """A filter feature whose single group value is that unhashable leaf."""
    return Feature(SFO_LEAF_FEATURE, Options(group={SFO_LEAF_KEY: leaf}))


def test_a_filter_feature_with_an_unhashable_option_value_keeps_its_hash() -> None:
    """Structural pin: the ownership copy must SHARE every option value, never copy it.

    _make_hashable falls back to repr() for an unhashable non-container leaf and the default repr
    embeds the object address, so swapping the copy for a deepcopy silently shifts the stored hash.
    """
    feature = _leaf_feature(_SfoUnhashableLeaf())
    global_filter = GlobalFilter()
    global_filter.add_filter(feature, FilterType.EQUAL, {"value": 1})
    stored = next(iter(global_filter.filters))
    assert hash(stored.filter_feature) == hash(feature), "the stored feature must keep the caller's hash"


def test_a_filter_feature_with_an_unhashable_option_value_still_dedupes() -> None:
    """The same structural pin from the dedup side: a value-equal re-add must not grow the set."""
    leaf = _SfoUnhashableLeaf()
    global_filter = GlobalFilter()
    global_filter.add_filter(_leaf_feature(leaf), FilterType.EQUAL, {"value": 1})
    global_filter.add_filter(_leaf_feature(leaf), FilterType.EQUAL, {"value": 1})
    assert len(global_filter.filters) == 1, "a value-equal re-add must deduplicate"
