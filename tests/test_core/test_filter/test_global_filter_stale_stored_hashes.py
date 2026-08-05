"""GlobalFilter's hash-keyed sets must stay usable when a stored filter's hash shifted.

record_probe merges a caller-owned set and CPython's set_merge copies that set's stored hashes
verbatim; rehash_stored_filters repairs a write that lands after storage.
"""

from __future__ import annotations

from uuid import uuid4

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.filter.single_filter import SingleFilter
from mloda.user import FilterType, GlobalFilter, Options

# The gfsh_ prefix keeps these names unique repo-wide.
GFSH_COLUMN = "gfsh_probe_col"
GFSH_RENAMED = "gfsh_probe_col_utc"
GFSH_HOST = "gfsh_host"
GFSH_LATE_KEY = "gfsh_late_key"


class GfshKeyGroup(FeatureGroup):
    """Inert key material for the hash-keyed sets: it never matches a feature."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False


def _declared() -> GlobalFilter:
    """record_probe records nothing while `filters` is empty, so one declared filter is the setup."""
    global_filter = GlobalFilter()
    global_filter.add_filter(GFSH_COLUMN, FilterType.MIN, {"value": 1})
    return global_filter


def _matched() -> SingleFilter:
    """A per-match copy as identify_matched_filters hands it to the engine."""
    return SingleFilter(GFSH_COLUMN, FilterType.MIN, {"value": 1})


def _record(global_filter: GlobalFilter, matched_filters: set[SingleFilter]) -> set[SingleFilter]:
    """Record one probe against a fresh key and return the set it was stored in."""
    host = FeatureName(GFSH_HOST)
    host_uuid = uuid4()
    global_filter.record_probe(GfshKeyGroup, host, host_uuid, matched_filters)
    return global_filter.probes[(GfshKeyGroup, host, host_uuid)]


def test_record_probe_stores_a_findable_filter_after_a_caller_side_rename() -> None:
    """Engine._add_filter_feature renames a matched filter feature before recording the set."""
    global_filter = _declared()
    matched = _matched()
    caller_set = {matched}

    matched.filter_feature.name = FeatureName(GFSH_RENAMED)

    stored = _record(global_filter, caller_set)
    assert matched not in caller_set, "the caller's set must be stale, else this pin proves nothing"
    assert matched in stored, "record_probe must store a findable filter, not the caller's stale hash"


def test_record_probe_stores_a_findable_filter_after_a_caller_side_options_rebind() -> None:
    """Intake rebinds a matched filter feature's options before recording the set."""
    global_filter = _declared()
    matched = _matched()
    caller_set = {matched}

    matched.filter_feature.options = Options(group={GFSH_LATE_KEY: "materialized"})

    stored = _record(global_filter, caller_set)
    assert matched not in caller_set, "the caller's set must be stale, else this pin proves nothing"
    assert matched in stored, "record_probe must store a findable filter, not the caller's stale hash"


def test_record_probe_keeps_a_healthy_filter_findable() -> None:
    """Guard: an unmutated filter is findable, so the pins above are not passing vacuously."""
    global_filter = _declared()
    matched = _matched()

    stored = _record(global_filter, {matched})
    assert matched in stored, "an unmutated filter must be findable in its own probes set"


def test_rehash_stored_filters_restores_membership_and_is_idempotent() -> None:
    """A filter mutated after storage is findable again after the rehash; a second call changes nothing."""
    global_filter = GlobalFilter()
    single_filter = _matched()
    global_filter.filters.add(single_filter)

    host = FeatureName(GFSH_HOST)
    host_uuid = uuid4()
    global_filter.add_filter_to_collection(GfshKeyGroup, host, single_filter)
    global_filter.record_probe(GfshKeyGroup, host, host_uuid, {single_filter})

    # Reachable after storage: an option value shared with a queue Feature is written in place later.
    single_filter.filter_feature.options.add_to_group(GFSH_LATE_KEY, "written after storage")
    assert single_filter not in global_filter.collection[(GfshKeyGroup, host)]
    assert single_filter not in global_filter.probes[(GfshKeyGroup, host, host_uuid)]

    global_filter.rehash_stored_filters()

    assert single_filter in global_filter.collection[(GfshKeyGroup, host)]
    assert single_filter in global_filter.probes[(GfshKeyGroup, host, host_uuid)]

    collection_snapshot = {key: set(value) for key, value in global_filter.collection.items()}
    probes_snapshot = {key: set(value) for key, value in global_filter.probes.items()}
    global_filter.rehash_stored_filters()
    assert global_filter.collection == collection_snapshot
    assert global_filter.probes == probes_snapshot
