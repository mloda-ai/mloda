"""Pin filter-feature intake ordering for an explicitly declared None option (#904).

Intake must run BEFORE ``Engine._add_filter_feature`` stores the matched ``SingleFilter``, because
it can rebind the filter feature's options (an explicit ``None`` reaches intake unfilled and is then
treated as absent: ``docs/docs/in_depth/property-mapping.md``). Rebinding an entry already inside a
set breaks it two ways: a group key shifts ``SingleFilter.__hash__``, and either category changes
``Feature.__eq__``, so a host reached twice stores the same filter twice.

The symptoms differ because ``set.__contains__`` recomputes the hash while ``set.__eq__`` compares
STORED hashes, so two equally stale sets still compare equal: the membership probe below catches the
group case, but a stale duplicate only aborts the run once two hosts hold unequal entry counts.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator, PropertySpec
from mloda.user import FeatureName, FilterType, GlobalFilter, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


# PROPERTY_MAPPING keys/defaults for the throwaway probes; the fen_ prefix keeps them unique to this module.
FEN_GRP_KEY = "fen_grp_key"
FEN_GRP_DEFAULT = "fen_grp_default_val"
FEN_CTX_KEY = "fen_ctx_key"
FEN_CTX_DEFAULT = "fen_ctx_default_val"
FEN_EN_KEY = "fen_explicit_none_key"
FEN_EN_DEFAULT = "fen_explicit_none_default_val"


class _Probe(NamedTuple):
    """One explicit-None variant: its declared key/spec plus the feature names unique to it."""

    key: str
    spec: PropertySpec
    context_key: bool  # where the explicit None is declared; the group half is the hash-shifting one
    root: str  # host feature the filter attaches to
    root_b: str  # second host of the same feature group, requested directly so it is reached once
    target: str  # filter feature; never requested, only reachable through the filter
    derived_a: str  # two derived features, both declaring root as their input feature
    derived_b: str


GRP_PROBE = _Probe(
    key=FEN_GRP_KEY,
    spec=PropertySpec("A group concrete default.", context=False, default=FEN_GRP_DEFAULT),
    context_key=False,
    root="fen_grp_root",
    root_b="fen_grp_root_b",
    target="fen_grp_target",
    derived_a="fen_grp_derived_a",
    derived_b="fen_grp_derived_b",
)

CTX_PROBE = _Probe(
    key=FEN_CTX_KEY,
    spec=PropertySpec("A context concrete default.", context=True, default=FEN_CTX_DEFAULT),
    context_key=True,
    root="fen_ctx_root",
    root_b="fen_ctx_root_b",
    target="fen_ctx_target",
    derived_a="fen_ctx_derived_a",
    derived_b="fen_ctx_derived_b",
)

EN_PROBE = _Probe(
    key=FEN_EN_KEY,
    spec=PropertySpec(
        "A group concrete default honoring an explicit None.",
        context=False,
        default=FEN_EN_DEFAULT,
        allow_explicit_none=True,
    ),
    context_key=False,
    root="fen_en_root",
    root_b="fen_en_root_b",
    target="fen_en_target",
    derived_a="fen_en_derived_a",
    derived_b="fen_en_derived_b",
)


def _explicit_none_options(probe: _Probe) -> Options:
    """A fresh Options declaring probe.key explicitly as None, in the spec's own category."""
    if probe.context_key:
        return Options(context={probe.key: None})
    return Options(group={probe.key: None})


def _make_host_fg(probe: _Probe) -> type[FeatureGroup]:
    """A throwaway root FeatureGroup serving the host and the filter feature, echoing its option view."""

    class FenFilterHostFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {probe.key: probe.spec}

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({probe.root, probe.root_b, probe.target})

        @classmethod
        def final_filters(cls) -> bool:
            # The payload is not filterable data: read features.filters inline instead of running
            # post-calculation row elimination against it.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            payload = {
                "names": sorted(str(f.name) for f in features.features),
                "observed": {str(f.name): f.options.get(probe.key) for f in features.features},
                "filter_count": len(features.filters) if features.filters else 0,
            }
            return {str(feature.name): [payload] for feature in features.features}

    return FenFilterHostFeatureGroup


def _make_derived_fg(probe: _Probe) -> type[FeatureGroup]:
    """A throwaway consumer serving two names that BOTH declare the host as their input feature, so
    the host is processed twice and the engine runs filter matching for it twice."""

    class FenDerivedFeatureGroup(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {probe.derived_a, probe.derived_b}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature(probe.root)}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(feature.name): [data[probe.root][0]] for feature in features.features}

    return FenDerivedFeatureGroup


def _summarize(global_filter: GlobalFilter, key: str) -> dict[str, dict[str, Any]]:
    """Plain-data view of the collection keyed by host feature name; the collection keys hold the
    throwaway feature group class, so only plain values leave this frame."""
    summary: dict[str, dict[str, Any]] = {}
    for (feature_group, host_name), stored in global_filter.collection.items():
        entries = list(stored)
        summary[str(host_name)] = {
            "feature_group": feature_group.get_class_name(),
            "entries": len(entries),
            # A stored entry whose hash shifted after insertion is no longer findable in its own set.
            "membership": [single in stored for single in entries],
            "filter_names": sorted(str(single.filter_feature.name) for single in entries),
            "stored_key_values": [single.filter_feature.options.get(key) for single in entries],
        }
    return summary


def _payload_rows(frames: list[Any], column: str) -> list[Any]:
    """Every row stored under ``column``, tolerant of columnar dict or list-of-row-dicts frames.
    Deliberately assert-free so the caller can drop the run's objects before judging the outcome."""
    rows: list[Any] = []
    for frame in frames:
        if isinstance(frame, dict):
            if column in frame:
                rows.extend(frame[column])
        else:
            rows.extend(row[column] for row in frame if column in row)
    return rows


def _run(probe: _Probe, via_derived: bool, second_host: bool = False) -> dict[str, Any]:
    """Run the probe under a global EQUAL filter whose filter feature declares probe.key as None.

    ``second_host`` requests probe.root_b as well, so one host of the group is reached once while the
    derived pair reaches probe.root twice. Returns the collection summary plus the payload row of the
    observed column. Every object referencing a throwaway class (the classes, the collector, the
    GlobalFilter whose collection keys hold the host class, the results) is deleted from THIS frame
    before the asserts below, so a failing assert cannot pin them into a traceback and trip the
    no-leak fixture on top of the real failure.
    """
    # The request carries the same explicit None as the filter feature: with allow_explicit_none
    # neither side is filled, which keeps host and filter feature in one FeatureSet. For the other
    # probes intake fills both sides anyway, so the request options make no difference there.
    feature_groups: set[type[FeatureGroup]] = {_make_host_fg(probe)}
    column = probe.root
    requested: list[Feature | str] = [Feature(probe.root, _explicit_none_options(probe))]
    if via_derived:
        feature_groups.add(_make_derived_fg(probe))
        column = probe.derived_a
        requested = [Feature(name, _explicit_none_options(probe)) for name in (probe.derived_a, probe.derived_b)]
    if second_host:
        requested.append(Feature(probe.root_b, _explicit_none_options(probe)))

    collector = PluginCollector.enabled_feature_groups(feature_groups)
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature(probe.target, _explicit_none_options(probe)), FilterType.EQUAL, {"value": 1})
    results = mloda.run_all(
        requested,
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
        global_filter=global_filter,
    )

    frames = list(results)
    summary = _summarize(global_filter, probe.key)
    rows = _payload_rows(frames, column)
    frames_repr = repr(frames)
    del feature_groups, collector, global_filter, results, frames, requested

    assert len(rows) == 1, f"expected exactly one payload row for {column}, got frames: {frames_repr}"
    payload = rows[0]
    assert isinstance(payload, dict), f"expected a payload dict for {column}, got: {payload!r}"
    return {"collection": summary, "payload": payload}


def _host_entry(observed: dict[str, Any], probe: _Probe) -> dict[str, Any]:
    """The collection entry recorded for the host feature; fails loudly when the filter never matched."""
    collection = observed["collection"]
    assert probe.root in collection, f"the filter must have matched and been collected: {collection!r}"
    entry = collection[probe.root]
    assert isinstance(entry, dict)
    # Identity only: the entry COUNT is what the dedup tests judge, so this shared guard must not pre-empt them.
    assert set(entry["filter_names"]) == {probe.target}, f"every collected filter must be the target feature: {entry!r}"
    return entry


def test_group_key_stored_filter_stays_findable_in_its_own_set() -> None:
    """The stored group-key filter stays findable in its own set (fails pre-fix: intake fills the
    declared default after insertion, so the entry's hash shifts and the set loses its own member)."""
    entry = _host_entry(_run(GRP_PROBE, via_derived=False), GRP_PROBE)
    assert entry["entries"] == 1, f"exactly one filter must be stored for the host feature: {entry!r}"
    assert entry["membership"] == [True], f"the stored filter must stay findable in its own set: {entry!r}"


def test_group_key_filter_is_stored_once_when_the_host_is_reached_twice() -> None:
    """A host reached by two consumers collects its group-key filter once (fails pre-fix: the rebound
    entry stops comparing equal, so the second pass lands a duplicate and the FeatureSet gets both)."""
    observed = _run(GRP_PROBE, via_derived=True)
    entry = _host_entry(observed, GRP_PROBE)
    assert entry["entries"] == 1, f"the same filter must be stored once for the host feature: {entry!r}"
    assert observed["payload"]["filter_count"] == 1, f"the FeatureSet must receive one filter: {observed['payload']!r}"


def test_group_key_two_hosts_with_unequal_reach_keep_one_filter_each() -> None:
    """Two hosts of one group, one reached twice and one once, keep a filter each and the run finishes.

    Fails pre-fix: the twice-reached host collects two entries against the once-reached host's one.
    The plan no longer rejects unequal sets, it attaches the deduplicated union, so the entry-count
    assertions are what guard the intake dedup now. Two equally stale sets still compare equal,
    which is why the symmetric dedup tests cannot reach this shape.
    """
    observed = _run(GRP_PROBE, via_derived=True, second_host=True)
    collection = observed["collection"]
    counts = {name: entry["entries"] for name, entry in collection.items()}
    assert counts == {GRP_PROBE.root: 1, GRP_PROBE.root_b: 1}, (
        f"each host must hold exactly one filter; the entry counts are what guard the dedup now: {collection!r}"
    )
    payload = observed["payload"]
    assert {GRP_PROBE.root, GRP_PROBE.root_b} <= set(payload["names"]), (
        f"both hosts must share one FeatureSet, else their filters are never compared: {payload!r}"
    )
    assert payload["filter_count"] == 1, f"the shared FeatureSet must receive one filter: {payload!r}"


def test_group_key_filter_feature_computes_with_the_materialized_default() -> None:
    """Guard, passes pre-fix: running intake earlier must not stop the fill, so the declared default
    still reaches both the computed filter feature and the stored entry (over-reaching-fix guard)."""
    observed = _run(GRP_PROBE, via_derived=False)
    payload = observed["payload"]
    assert GRP_PROBE.target in payload["names"], f"the filter must attach its target feature: {payload!r}"
    assert payload["observed"][GRP_PROBE.target] == FEN_GRP_DEFAULT, (
        f"the filter feature must compute with the materialized default: {payload!r}"
    )
    entry = _host_entry(observed, GRP_PROBE)
    assert entry["stored_key_values"] == [FEN_GRP_DEFAULT], f"the collection must store the effective view: {entry!r}"


def test_context_key_stored_filter_stays_findable_in_its_own_set() -> None:
    """Guard, passes pre-fix: ``Options.__hash__`` covers group only, so a context fill leaves the
    stored hash intact; the dedup twin below is the discriminating half of the context case."""
    entry = _host_entry(_run(CTX_PROBE, via_derived=False), CTX_PROBE)
    assert entry["entries"] == 1, f"exactly one filter must be stored for the host feature: {entry!r}"
    assert entry["membership"] == [True], f"the stored filter must stay findable in its own set: {entry!r}"


def test_context_key_filter_is_stored_once_when_the_host_is_reached_twice() -> None:
    """A host reached by two consumers collects its context-key filter once (fails pre-fix:
    ``Feature.__eq__`` compares ``options.context``, so the rebound entry stops comparing equal)."""
    observed = _run(CTX_PROBE, via_derived=True)
    entry = _host_entry(observed, CTX_PROBE)
    assert entry["entries"] == 1, f"the same filter must be stored once for the host feature: {entry!r}"
    assert observed["payload"]["filter_count"] == 1, f"the FeatureSet must receive one filter: {observed['payload']!r}"


def test_opted_in_explicit_none_filter_is_never_rebound() -> None:
    """Guard, passes pre-fix: ``allow_explicit_none=True`` honors the None, so nothing is rebound and
    the None survives to compute time; a fix that dropped the explicit None would break this."""
    observed = _run(EN_PROBE, via_derived=False)
    entry = _host_entry(observed, EN_PROBE)
    assert entry["entries"] == 1, f"exactly one filter must be stored for the host feature: {entry!r}"
    assert entry["membership"] == [True], f"the stored filter must stay findable in its own set: {entry!r}"
    assert entry["stored_key_values"] == [None], f"the opted-in explicit None must survive intake: {entry!r}"
    payload = observed["payload"]
    assert payload["observed"][EN_PROBE.target] is None, (
        f"the filter feature must compute with the explicit None, not the default: {payload!r}"
    )


def test_opted_in_explicit_none_filter_is_stored_once_when_the_host_is_reached_twice() -> None:
    """Guard, passes pre-fix: without a rebind both passes store the same filter into one entry."""
    observed = _run(EN_PROBE, via_derived=True)
    entry = _host_entry(observed, EN_PROBE)
    assert entry["entries"] == 1, f"the same filter must be stored once for the host feature: {entry!r}"
    assert observed["payload"]["filter_count"] == 1, f"the FeatureSet must receive one filter: {observed['payload']!r}"
