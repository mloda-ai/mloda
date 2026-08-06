"""A filter feature must not inherit a NON_FORWARDED_KEYS option from the feature it attaches to.

Importing in_features hands the filter the host's dependency declaration and shares the host's
mutable Feature value into the stored SingleFilter's hash.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import NON_FORWARDED_KEYS, Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.filter.single_filter import SingleFilter
from mloda.user import FilterType, GlobalFilter

# The nfk_ prefix keeps these names unique repo-wide.
NFK_FILTER_COL = "nfk_filter_col"
NFK_HOST_ONE = "nfk_host_one"
NFK_HOST_TWO = "nfk_host_two"
NFK_LEAF = "nfk_leaf"
NFK_VARIANT = "nfk_variant"
NFK_ORDINARY = "nfk_ordinary"
NFK_HOST_VALUE = "nfk_host_value"
NFK_OWN_VALUE = "nfk_own_value"

_BLOCKED_KEYS = sorted(NON_FORWARDED_KEYS)


class NfkHostGroup(FeatureGroup):
    """Matches the filter column only, so identify_matched_filters attaches exactly one filter."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == NFK_FILTER_COL


def _stranded(stored_sets: Iterable[set[SingleFilter]]) -> list[str]:
    """Names of the stored filters their own set can no longer find."""
    return [single.name for stored in stored_sets for single in stored if single not in stored]


def _host(name: str, variant: int, shared: Feature) -> Feature:
    return Feature(name, Options(group={DefaultOptionKeys.in_features.value: shared, NFK_VARIANT: variant}))


def _declared() -> GlobalFilter:
    global_filter = GlobalFilter()
    global_filter.add_filter(NFK_FILTER_COL, FilterType.MIN, {"value": 1})
    return global_filter


def _store_matches(global_filter: GlobalFilter, host: Feature) -> set[SingleFilter]:
    """Match and store exactly as Engine._add_filter_feature does."""
    matched = global_filter.identify_matched_filters(NfkHostGroup, host)
    for match in matched:
        # The stored filter takes its own Feature via Feature.__copy__: one level deep, values shared.
        match.filter_feature = match.handle_filter_feature(match.filter_feature)
        global_filter.add_filter_to_collection(NfkHostGroup, host.name, match)
    global_filter.record_probe(NfkHostGroup, host.name, host.uuid, matched)
    return matched


def test_the_parametrized_key_list_is_populated() -> None:
    """Guard: an emptied NON_FORWARDED_KEYS would silently skip every case below instead of failing it."""
    assert DefaultOptionKeys.in_features in NON_FORWARDED_KEYS


@pytest.mark.parametrize("blocked", _BLOCKED_KEYS)
def test_unify_options_skips_a_blocked_group_key(blocked: str) -> None:
    unified = GlobalFilter().unify_options(
        Options(group={blocked: NFK_HOST_VALUE, NFK_ORDINARY: 1}),
        Options(),
    )

    assert blocked not in unified, f"'{blocked}' must not be imported from the host: {unified}"
    assert unified.group == {NFK_ORDINARY: 1}, f"every other host group key must still arrive: {unified.group}"


@pytest.mark.parametrize("blocked", _BLOCKED_KEYS)
def test_unify_options_skips_a_blocked_context_key(blocked: str) -> None:
    unified = GlobalFilter().unify_options(
        Options(context={blocked: NFK_HOST_VALUE, NFK_ORDINARY: 1}),
        Options(),
    )

    assert blocked not in unified, f"'{blocked}' must not be imported from the host: {unified}"
    assert unified.context == {NFK_ORDINARY: 1}, f"every other host context key must still arrive: {unified.context}"


@pytest.mark.parametrize("blocked", _BLOCKED_KEYS)
def test_unify_options_keeps_a_blocked_key_the_filter_feature_declared_itself(blocked: str) -> None:
    """Guard: skipping the import must not turn into rewriting the filter feature's own declaration."""
    unified = GlobalFilter().unify_options(
        Options(group={blocked: NFK_HOST_VALUE}),
        Options(group={blocked: NFK_OWN_VALUE}),
    )

    assert unified.group == {blocked: NFK_OWN_VALUE}, f"a declared value is never rewritten: {unified.group}"


def test_a_matched_filter_does_not_inherit_the_host_input_feature_declaration() -> None:
    """The host's in_features is its own dependency declaration, not the filter feature's."""
    global_filter = _declared()

    matched = global_filter.identify_matched_filters(NfkHostGroup, _host(NFK_HOST_ONE, 1, Feature(NFK_LEAF)))

    names = {single.name for single in matched}
    assert names == {NFK_FILTER_COL}, f"guard: exactly one filter must match the host: {names!r}"
    options = next(iter(matched)).filter_feature.options
    assert DefaultOptionKeys.in_features not in options, (
        f"the filter feature must not carry the host's input-feature list: {options}"
    )
    assert options.group == {NFK_VARIANT: 1}, f"every other host group key must still arrive: {options.group}"


def test_a_restamped_shared_input_feature_keeps_stored_filters_findable_without_the_repair() -> None:
    """Two hosts share one input Feature; re-stamping it must not strand an already stored filter."""
    global_filter = _declared()
    shared = Feature(NFK_LEAF)

    _store_matches(global_filter, _host(NFK_HOST_ONE, 1, shared))
    _store_matches(global_filter, _host(NFK_HOST_TWO, 2, shared))

    hosts = {str(name) for _group, name in global_filter.collection}
    assert hosts == {NFK_HOST_ONE, NFK_HOST_TWO}, f"guard: the filter must attach to both hosts: {hosts!r}"

    # What Features.build_feature_collection does to a shared input Feature while a later host resolves.
    shared.child_options = Options(group={NFK_VARIANT: 2})

    assert _stranded(global_filter.collection.values()) == [], "a collected filter must be findable in its own set"
    assert _stranded(global_filter.probes.values()) == [], "a probed filter must be findable in its own set"
