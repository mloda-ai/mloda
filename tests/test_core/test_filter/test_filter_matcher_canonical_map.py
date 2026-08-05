"""Issue #728: map GlobalFilter.identify_matched_filters against the canonical resolver, axis by axis:
scope, framework-pin cardinality, the capability hook, domain defaulting, links, and option enrichment.
Truthiness (#927) and raise containment (#899) live in the sibling suites. Probe classes live inside factory
functions and are dropped before any assert runs, so a failing assert never trips the no-leak conftest fixture.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, TypeVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import ComputeFrameworkPinError, IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_types import EvaluationResult
from mloda.user import Feature, FeatureName, FilterType, GlobalFilter, JoinSpec, Link, Options, SingleFilter
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


HOST_FEATURE = "cmap_host_feat_728"  # the resolved feature the filters are matched against
FILTER_FEATURE = "cmap_filter_feat_728"  # the declared filter feature both seams are asked about

PLAIN_CLASS_NAME = "PlainMatcherFG728"
GROUP_DOMAIN_CLASS_NAME = "GroupDomainMatcherFG728"
CAPABILITY_CLASS_NAME = "CapabilityRejectMatcherFG728"
OWN_INDEX_CLASS_NAME = "OwnIndexMatcherFG728"

MISSING_SCOPE_728 = "CmapNoSuchScope728"  # a scope string naming no accessible class
SCOPE_REASON = "outside the requested feature group scope"
PIN_MESSAGE_PART = "more than one compute framework"
CAPABILITY_REASON_PART = "supports_compute_framework rejected"
LINKS_REASON = "no index column matches the run's links"

GROUP_DOMAIN_728 = "cmap_group_domain_728"  # declared by the group-domain probe
FEATURE_DOMAIN_728 = "cmap_feature_domain_728"  # carried by the resolved feature
OTHER_DOMAIN_728 = "cmap_other_domain_728"  # matches neither
DEFAULT_DOMAIN_NAME = "default_domain"

OWN_INDEX_728 = "cmap_own_idx_728"
LINK_LEFT_728 = "cmap_link_left_728"
LINK_RIGHT_728 = "cmap_link_right_728"

SHARED_KEY_728 = "cmap_shared_key_728"  # declared by host and filter feature with different values
GROUP_ONLY_KEY_728 = "cmap_group_only_key_728"
CONTEXT_ONLY_KEY_728 = "cmap_context_only_key_728"
FEATURE_VALUE = "feature_value"
FILTER_VALUE = "filter_value"
GROUP_ONLY_VALUE = "group_only_value"
CONTEXT_ONLY_VALUE = "context_only_value"

T = TypeVar("T")

# A factory handing back the throwaway class and a reader for its call counter, so no drive types the class.
_CounterFactory = Callable[[], tuple[type[FeatureGroup], Callable[[], int]]]
_ObservedOptions = tuple[tuple[tuple[str, str], ...], ...]


def _capture(call: Callable[[], T]) -> tuple[T | None, str | None]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (characterization probe: the absence of an escape is itself pinned)
        return None, f"{type(exc).__name__}: {exc}"


def _single(filter_feature: Feature | str = FILTER_FEATURE) -> SingleFilter:
    """A minimal EQUAL filter on the module's filter feature, or on a caller-built Feature."""
    return SingleFilter(filter_feature, FilterType.EQUAL, {"value": 1})


def _make_plain_matcher_fg() -> type[FeatureGroup]:
    """A throwaway group matching both module names through the default criteria."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class PlainMatcherFG728(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

    return PlainMatcherFG728


def _make_group_domain_fg() -> type[FeatureGroup]:
    """The same matcher declaring a non-default group domain."""
    gc.collect()

    class GroupDomainMatcherFG728(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(GROUP_DOMAIN_728)

    return GroupDomainMatcherFG728


def _make_capability_reject_fg() -> tuple[type[FeatureGroup], Callable[[], int]]:
    """A throwaway matcher whose capability hook counts its calls and rejects every framework."""
    gc.collect()

    class CapabilityRejectMatcherFG728(FeatureGroup):
        calls: ClassVar[int] = 0

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

        @classmethod
        def supports_compute_framework(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            compute_framework: type[ComputeFramework],
        ) -> bool:
            cls.calls += 1
            return False

    return CapabilityRejectMatcherFG728, lambda: CapabilityRejectMatcherFG728.calls


def _make_own_index_fg() -> tuple[type[FeatureGroup], Callable[[], int]]:
    """A throwaway matcher whose index hook counts its calls and declares one own index column."""
    gc.collect()

    class OwnIndexMatcherFG728(FeatureGroup):
        calls: ClassVar[int] = 0

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

        @classmethod
        def index_columns(cls) -> list[Index] | None:
            cls.calls += 1
            return [Index((OWN_INDEX_728,))]

    return OwnIndexMatcherFG728, lambda: OwnIndexMatcherFG728.calls


def _make_option_recorder_fg() -> tuple[type[FeatureGroup], Callable[[], _ObservedOptions]]:
    """A throwaway matcher recording the option view its criteria hook observes for FILTER_FEATURE."""
    gc.collect()

    class OptionRecorderMatcherFG728(FeatureGroup):
        observed: ClassVar[list[tuple[tuple[str, str], ...]]] = []

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                keys = (SHARED_KEY_728, GROUP_ONLY_KEY_728, CONTEXT_ONLY_KEY_728)
                cls.observed.append(tuple((key, str(options.get(key))) for key in keys))
            return str(feature_name) in cls.feature_names_supported()

    return OptionRecorderMatcherFG728, lambda: tuple(OptionRecorderMatcherFG728.observed)


@dataclass(frozen=True)
class _MatchingSnapshot:
    """Plain-data readout of one identify_matched_filters call. Holds no class and no filter object."""

    escaped: str | None
    names: tuple[str, ...]
    calls: int = 0
    scopes: tuple[str, ...] = ()


def _drive_scoped_matching() -> _MatchingSnapshot:
    """Match one filter declared with a scope naming no class."""
    fg = _make_plain_matcher_fg()
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature(FILTER_FEATURE, feature_group=MISSING_SCOPE_728), FilterType.EQUAL, {"value": 1})
    matched = None
    try:
        matched, escaped = _capture(partial(global_filter.identify_matched_filters, fg, Feature(HOST_FEATURE), None))
        names: tuple[str, ...] = ()
        scopes: tuple[str, ...] = ()
        if matched is not None:
            names = tuple(sorted(single.name for single in matched))
            scopes = tuple(sorted(str(single.filter_feature.feature_group_scope) for single in matched))
        return _MatchingSnapshot(escaped=escaped, names=names, scopes=scopes)
    finally:
        del fg, global_filter, matched
        gc.collect()


def _drive_two_pin_matching() -> _MatchingSnapshot:
    """Match one two-pin filter against a PythonDict-pinned host."""
    fg = _make_plain_matcher_fg()
    global_filter = GlobalFilter()
    pinned = Feature(FILTER_FEATURE)
    pinned.compute_frameworks = {PandasDataFrame, PyArrowTable}
    global_filter.add_filter(pinned, FilterType.EQUAL, {"value": 1})
    host = Feature(HOST_FEATURE)
    host.compute_frameworks = {PythonDictFramework}
    matched = None
    try:
        matched, escaped = _capture(partial(global_filter.identify_matched_filters, fg, host, None))
        return _MatchingSnapshot(
            escaped=escaped,
            names=() if matched is None else tuple(sorted(single.name for single in matched)),
        )
    finally:
        del fg, global_filter, matched
        gc.collect()


def _drive_matching_counted(make: _CounterFactory, pin_host: bool) -> _MatchingSnapshot:
    """Match one unpinned registered filter against HOST_FEATURE and read the probe's call counter."""
    fg, read_calls = make()
    global_filter = GlobalFilter()
    global_filter.add_filter(FILTER_FEATURE, FilterType.EQUAL, {"value": 1})
    host = Feature(HOST_FEATURE)
    if pin_host:
        host.compute_frameworks = {PythonDictFramework}
    matched = None
    try:
        matched, escaped = _capture(partial(global_filter.identify_matched_filters, fg, host, None))
        return _MatchingSnapshot(
            escaped=escaped,
            names=() if matched is None else tuple(sorted(single.name for single in matched)),
            calls=read_calls(),
        )
    finally:
        del fg, read_calls, global_filter, matched
        gc.collect()


@dataclass(frozen=True)
class _CanonicalSnapshot:
    """Plain-data readout of one evaluate() pass. Holds no class and no Elimination object."""

    escaped: str | None
    identified: tuple[str, ...]
    eliminations: tuple[tuple[str, str, str], ...]
    calls: int = 0


def _canonical_snapshot(result: EvaluationResult | None, escaped: str | None, calls: int = 0) -> _CanonicalSnapshot:
    """Fold one evaluate() outcome to plain data."""
    if result is None:
        return _CanonicalSnapshot(escaped=escaped, identified=(), eliminations=(), calls=calls)
    return _CanonicalSnapshot(
        escaped=escaped,
        identified=tuple(sorted(g.get_class_name() for g in result.identified)),
        eliminations=tuple(
            sorted((g.get_class_name(), str(e.stage), str(e.reason)) for g, e in result.eliminations.items())
        ),
        calls=calls,
    )


def _drive_canonical(build: Callable[[], type[FeatureGroup]], scope: str | None = None) -> _CanonicalSnapshot:
    """Evaluate FILTER_FEATURE (optionally scope-constrained) against the probe alone."""
    fg = build()
    plugins: FeatureGroupEnvironmentMapping = {fg: {PythonDictFramework}}
    result = None
    try:
        feature = Feature(FILTER_FEATURE, feature_group=scope)
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None))
        snapshot = _canonical_snapshot(result, escaped)
        del result
        result = None
        return snapshot
    finally:
        del fg, plugins, result
        gc.collect()


def _drive_canonical_counted(make: _CounterFactory, with_links: bool) -> _CanonicalSnapshot:
    """Evaluate FILTER_FEATURE against a counting probe, optionally under links naming only unknown columns."""
    fg, read_calls = make()
    plugins: FeatureGroupEnvironmentMapping = {fg: {PythonDictFramework}}
    links = {Link("inner", JoinSpec(fg, (LINK_LEFT_728,)), JoinSpec(fg, (LINK_RIGHT_728,)))} if with_links else None
    result = None
    try:
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, Feature(FILTER_FEATURE), plugins, links))
        snapshot = _canonical_snapshot(result, escaped, read_calls())
        del result
        result = None
        return snapshot
    finally:
        del fg, read_calls, plugins, links, result
        gc.collect()


def _drive_canonical_two_pin_message() -> str:
    """The canonical seam validates the pin before matching; hand back the raise's message as text."""
    fg = _make_plain_matcher_fg()
    plugins: FeatureGroupEnvironmentMapping = {fg: {PythonDictFramework}}
    feature = Feature(FILTER_FEATURE)
    feature.compute_frameworks = {PandasDataFrame, PyArrowTable}
    try:
        with pytest.raises(ComputeFrameworkPinError) as excinfo:
            IdentifyFeatureGroupClass.evaluate(feature, plugins, None)
        message = str(excinfo.value)
        # The retained traceback pins evaluate's frame and with it the probe class; drop it before returning.
        del excinfo
        return message
    finally:
        del fg, plugins, feature
        gc.collect()


def _domain_name_of(filter_feature: Feature) -> str | None:
    """The filter feature's domain name, or None when it carries no Domain."""
    domain = filter_feature.domain
    return domain.name if isinstance(domain, Domain) else None


@dataclass(frozen=True)
class _DomainSnapshot:
    """Plain-data readout of one GlobalFilter.domain call. Holds no class and no Domain object."""

    escaped: str | None
    verdict: bool | None
    domain_after: str | None


def _drive_domain(
    build: Callable[[], type[FeatureGroup]], filter_domain: str | None, feature_domain: str | None
) -> _DomainSnapshot:
    """Call GlobalFilter.domain once and read the verdict plus the (possibly adopted) filter-feature domain."""
    fg = build()
    global_filter = GlobalFilter()
    single = _single(Feature(FILTER_FEATURE, domain=filter_domain))
    requested = None if feature_domain is None else Domain(feature_domain)
    try:
        verdict, escaped = _capture(partial(global_filter.domain, single, requested, fg))
        return _DomainSnapshot(
            escaped=escaped,
            verdict=verdict,
            domain_after=_domain_name_of(single.filter_feature),
        )
    finally:
        del fg, global_filter, single
        gc.collect()


@dataclass(frozen=True)
class _DomainAdoptionSnapshot:
    """Plain-data readout of one group-domain matching pass. Holds no class and no filter object."""

    escaped: str | None
    names: tuple[str, ...]
    matched_domains: tuple[str | None, ...]
    stored_domain: str | None


def _drive_group_domain_adoption() -> _DomainAdoptionSnapshot:
    """Match a domainless filter and host on the group-domain probe; adoption may only touch the matched copy."""
    fg = _make_group_domain_fg()
    global_filter = GlobalFilter()
    global_filter.add_filter(FILTER_FEATURE, FilterType.EQUAL, {"value": 1})
    matched = None
    try:
        matched, escaped = _capture(partial(global_filter.identify_matched_filters, fg, Feature(HOST_FEATURE), None))
        names: tuple[str, ...] = ()
        matched_domains: tuple[str | None, ...] = ()
        if matched is not None:
            names = tuple(sorted(single.name for single in matched))
            matched_domains = tuple(_domain_name_of(single.filter_feature) for single in matched)
        return _DomainAdoptionSnapshot(
            escaped=escaped,
            names=names,
            matched_domains=matched_domains,
            stored_domain=_domain_name_of(next(iter(global_filter.filters)).filter_feature),
        )
    finally:
        del fg, global_filter, matched
        gc.collect()


@dataclass(frozen=True)
class _EnrichmentSnapshot:
    """Plain-data readout of one enriched matching pass. Holds no class and no filter object."""

    escaped: str | None
    names: tuple[str, ...]
    matched_group: tuple[tuple[str, str], ...]
    matched_context: tuple[tuple[str, str], ...]
    observed: _ObservedOptions
    stored_group: tuple[tuple[str, str], ...]
    stored_context: tuple[tuple[str, str], ...]


def _options_as_pairs(options: Options) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """(group, context) as sorted (key, str(value)) pairs."""
    group = tuple(sorted((str(key), str(value)) for key, value in options.group.items()))
    context = tuple(sorted((str(key), str(value)) for key, value in options.context.items()))
    return group, context


def _drive_enriched_matching(filter_options: Options | None = None) -> _EnrichmentSnapshot:
    """Match the filter feature, declaring the given options (default: the shared key in group), against the host."""
    fg, read_observed = _make_option_recorder_fg()
    global_filter = GlobalFilter()
    declared = Options(group={SHARED_KEY_728: FILTER_VALUE}) if filter_options is None else filter_options
    global_filter.add_filter(Feature(FILTER_FEATURE, declared), FilterType.EQUAL, {"value": 1})
    host = Feature(
        HOST_FEATURE,
        Options(
            group={SHARED_KEY_728: FEATURE_VALUE, GROUP_ONLY_KEY_728: GROUP_ONLY_VALUE},
            context={CONTEXT_ONLY_KEY_728: CONTEXT_ONLY_VALUE},
        ),
    )
    matched = None
    try:
        matched, escaped = _capture(partial(global_filter.identify_matched_filters, fg, host, None))
        names: tuple[str, ...] = ()
        matched_group: tuple[tuple[str, str], ...] = ()
        matched_context: tuple[tuple[str, str], ...] = ()
        for single in matched or ():
            names = (*names, single.name)
            matched_group, matched_context = _options_as_pairs(single.filter_feature.options)
        stored_group, stored_context = _options_as_pairs(next(iter(global_filter.filters)).filter_feature.options)
        return _EnrichmentSnapshot(
            escaped=escaped,
            names=tuple(sorted(names)),
            matched_group=matched_group,
            matched_context=matched_context,
            observed=read_observed(),
            stored_group=stored_group,
            stored_context=stored_context,
        )
    finally:
        del fg, read_observed, global_filter, matched
        gc.collect()


@dataclass(frozen=True)
class _CanonicalObservationSnapshot:
    """Plain-data readout of the options view one evaluate() pass showed the recorder hook."""

    escaped: str | None
    observed: _ObservedOptions


def _drive_canonical_declared_options() -> _CanonicalObservationSnapshot:
    """Evaluate the option-declaring filter feature alone; the canonical seam has no host to enrich from."""
    fg, read_observed = _make_option_recorder_fg()
    plugins: FeatureGroupEnvironmentMapping = {fg: {PythonDictFramework}}
    feature = Feature(FILTER_FEATURE, Options(group={SHARED_KEY_728: FILTER_VALUE}))
    result = None
    try:
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None))
        snapshot = _CanonicalObservationSnapshot(escaped=escaped, observed=read_observed())
        del result
        result = None
        return snapshot
    finally:
        del fg, read_observed, plugins, result
        gc.collect()


class TestScopeGateAbsence:
    """The filter seam never reads feature_group_scope; the canonical resolver eliminates on it at "scope"."""

    def test_the_filter_seam_attaches_a_filter_scoped_to_no_class(self) -> None:
        snapshot = _drive_scoped_matching()

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.names == (FILTER_FEATURE,), f"the scope must not detach the filter, got: {snapshot.names}"

    def test_the_matched_copy_still_carries_the_unread_scope(self) -> None:
        """The scope survives SingleFilter's copy and the per-match deepcopy: present, just never read."""
        snapshot = _drive_scoped_matching()

        assert snapshot.scopes == (MISSING_SCOPE_728,), f"the scope must ride along unread, got: {snapshot.scopes}"

    def test_the_canonical_seam_eliminates_the_same_scope_at_the_scope_gate(self) -> None:
        snapshot = _drive_canonical(_make_plain_matcher_fg, scope=MISSING_SCOPE_728)

        assert snapshot.escaped is None, f"nothing may cross evaluate: {snapshot.escaped}"
        assert snapshot.identified == (), f"a scope naming no class must win nothing, got: {snapshot.identified}"
        assert snapshot.eliminations == ((PLAIN_CLASS_NAME, "scope", SCOPE_REASON),), (
            f"exactly one scope elimination, got: {snapshot.eliminations}"
        )


class TestFrameworkPinCardinality:
    """GlobalFilter.compute_framework reads one arbitrary pinned element; the canonical seam validates first."""

    def test_an_unpinned_filter_feature_adopts_the_features_frameworks(self) -> None:
        """Enrichment control: no pin on the filter side means the feature's pin is written onto the filter."""
        global_filter = GlobalFilter()
        single = _single()
        feat = Feature(HOST_FEATURE)
        feat.compute_frameworks = {PythonDictFramework}

        verdict = global_filter.compute_framework(single, feat)

        assert verdict is True, "an unpinned filter feature must accept the feature's framework"
        assert single.filter_feature.compute_frameworks == {PythonDictFramework}, "the feature's pin is adopted"
        assert single.filter_feature.compute_frameworks is feat.compute_frameworks, (
            "adoption shares the feature's set object, not a copy"
        )

    def test_a_two_pin_filter_is_judged_by_the_first_yielded_set_element(self) -> None:
        """No cardinality validation: only the first-yielded pin decides; membership alone loses."""
        global_filter = GlobalFilter()
        single = _single()
        pins: set[type[ComputeFramework]] = {PandasDataFrame, PyArrowTable}
        single.filter_feature.compute_frameworks = pins
        # Derived from the set itself: iteration over the same unmutated set object is stable in-process, and
        # which element comes first is not pinned, so the pair of verdicts below is deterministic everywhere.
        first = next(iter(pins))
        other = next(pin for pin in pins if pin is not first)
        feat_first = Feature(HOST_FEATURE)
        feat_first.compute_frameworks = {first}
        feat_other = Feature(HOST_FEATURE)
        feat_other.compute_frameworks = {other}

        first_verdict = global_filter.compute_framework(single, feat_first)
        other_verdict = global_filter.compute_framework(single, feat_other)

        assert first_verdict is True, f"the first-yielded pin must match its own framework, got: {first_verdict}"
        assert other_verdict is False, f"a member the gate never reads must lose, got: {other_verdict}"

    def test_a_two_pin_filter_against_a_third_framework_says_no(self) -> None:
        global_filter = GlobalFilter()
        single = _single()
        single.filter_feature.compute_frameworks = {PandasDataFrame, PyArrowTable}
        feat = Feature(HOST_FEATURE)
        feat.compute_frameworks = {PythonDictFramework}

        verdict = global_filter.compute_framework(single, feat)

        assert verdict is False, "neither pinned framework equals the feature's, whichever the set yields first"
        assert single.filter_feature.compute_frameworks == {PandasDataFrame, PyArrowTable}, "the pin is not rewritten"

    def test_the_flow_detaches_a_mismatched_two_pin_quietly(self) -> None:
        snapshot = _drive_two_pin_matching()

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.names == (), f"a mismatched two-pin filter must not attach, got: {snapshot.names}"

    def test_the_canonical_seam_rejects_the_same_two_pin_outright(self) -> None:
        """Validate-before-matching itself is pinned in
        tests/test_core/test_prepare/test_identify_feature_group_evaluation_seam.py."""
        message = _drive_canonical_two_pin_message()

        assert PIN_MESSAGE_PART in message, f"the raise must name the misuse: {message}"
        assert FILTER_FEATURE in message, f"the raise must name the pinned feature: {message}"


class TestCapabilityHookAbsence:
    """The filter path never consults supports_compute_framework; the canonical seam splits frameworks on it."""

    def test_the_filter_seam_attaches_without_asking_the_hook(self) -> None:
        snapshot = _drive_matching_counted(_make_capability_reject_fg, pin_host=True)

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.names == (FILTER_FEATURE,), f"a rejecting hook must not detach here, got: {snapshot.names}"
        assert snapshot.calls == 0, f"identify_matched_filters must never consult the hook, got {snapshot.calls} calls"

    def test_the_canonical_seam_asks_the_hook_and_eliminates_at_capability(self) -> None:
        snapshot = _drive_canonical_counted(_make_capability_reject_fg, with_links=False)

        assert snapshot.escaped is None, f"nothing may cross evaluate: {snapshot.escaped}"
        assert snapshot.identified == (), f"an all-rejected candidate must win nothing, got: {snapshot.identified}"
        assert len(snapshot.eliminations) == 1, f"exactly one near-miss, got: {snapshot.eliminations}"
        name, stage, reason = snapshot.eliminations[0]
        assert name == CAPABILITY_CLASS_NAME, f"the near-miss must name the candidate, got: {name}"
        assert stage == "capability", f"the empty supported split owns the reason, got stage: {stage}"
        assert CAPABILITY_REASON_PART in reason, f"the reason must name the rejecting hook: {reason}"
        assert snapshot.calls >= 1, "evaluate must have consulted the hook"


class TestDomainDefaulting:
    """GlobalFilter.domain: the verdict per branch plus the domain the gate may write onto the filter copy."""

    @pytest.mark.parametrize(
        ("filter_domain", "feature_domain", "group_has_domain", "expected_verdict", "expected_after"),
        [
            pytest.param(None, None, False, True, None, id="D1_untouched"),
            pytest.param(None, None, True, True, GROUP_DOMAIN_728, id="D2_group_adopted"),
            pytest.param(None, FEATURE_DOMAIN_728, True, True, FEATURE_DOMAIN_728, id="D3_feature_outranks"),
            pytest.param(GROUP_DOMAIN_728, None, True, True, GROUP_DOMAIN_728, id="D4_filter_equals_group"),
            pytest.param(OTHER_DOMAIN_728, None, True, False, OTHER_DOMAIN_728, id="D5_filter_diverges"),
            pytest.param(DEFAULT_DOMAIN_NAME, None, False, True, DEFAULT_DOMAIN_NAME, id="D6_default_equality"),
            pytest.param(OTHER_DOMAIN_728, None, False, False, OTHER_DOMAIN_728, id="D6b_filter_vs_default_group"),
            pytest.param(FEATURE_DOMAIN_728, FEATURE_DOMAIN_728, False, True, FEATURE_DOMAIN_728, id="D7a_equal"),
            pytest.param(OTHER_DOMAIN_728, FEATURE_DOMAIN_728, False, False, OTHER_DOMAIN_728, id="D7b_diverging"),
        ],
    )
    def test_the_domain_branch_table(
        self,
        filter_domain: str | None,
        feature_domain: str | None,
        group_has_domain: bool,
        expected_verdict: bool,
        expected_after: str | None,
    ) -> None:
        build = _make_group_domain_fg if group_has_domain else _make_plain_matcher_fg
        snapshot = _drive_domain(build, filter_domain, feature_domain)

        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.domain: {snapshot.escaped}"
        assert snapshot.verdict is expected_verdict, f"wrong verdict for this branch, got: {snapshot.verdict}"
        assert snapshot.domain_after == expected_after, f"wrong filter domain after the gate: {snapshot.domain_after}"

    def test_the_canonical_domain_gate_ignores_the_group_domain_for_a_domainless_request(self) -> None:
        """Differential to D5: only the filter seam consults the group domain when the request carries none."""
        filter_side = _drive_domain(_make_group_domain_fg, OTHER_DOMAIN_728, None)
        canonical = _drive_canonical(_make_group_domain_fg)

        assert filter_side.escaped is None, f"nothing may cross GlobalFilter.domain: {filter_side.escaped}"
        assert filter_side.verdict is False, "the filter seam gates a domain-carrying filter on the group domain"
        assert canonical.escaped is None, f"nothing may cross evaluate: {canonical.escaped}"
        assert canonical.identified == (GROUP_DOMAIN_CLASS_NAME,), (
            f"the canonical gate passes any candidate for a domainless request, got: {canonical.identified}"
        )
        assert canonical.eliminations == (), f"no elimination may be recorded, got: {canonical.eliminations}"

    def test_the_flow_adopts_the_group_domain_onto_the_matched_copy_only(self) -> None:
        """D2 through identify_matched_filters: adoption lands on the deepcopy, never on the stored filter."""
        snapshot = _drive_group_domain_adoption()

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.names == (FILTER_FEATURE,), f"the domainless filter must attach, got: {snapshot.names}"
        assert snapshot.matched_domains == (GROUP_DOMAIN_728,), (
            f"the matched copy must adopt the group domain, got: {snapshot.matched_domains}"
        )
        assert snapshot.stored_domain is None, (
            f"the stored original must stay domainless, got: {snapshot.stored_domain}"
        )


class TestLinksSkip:
    """identify_matched_filters has no links parameter; the canonical links gate reads the index hooks."""

    def test_the_filter_seam_attaches_without_reading_index_columns(self) -> None:
        snapshot = _drive_matching_counted(_make_own_index_fg, pin_host=False)

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.names == (FILTER_FEATURE,), f"the filter must attach without a links gate: {snapshot.names}"
        assert snapshot.calls == 0, f"identify_matched_filters must never read index hooks, got {snapshot.calls} calls"

    def test_the_canonical_seam_reads_the_hook_and_eliminates_at_links(self) -> None:
        snapshot = _drive_canonical_counted(_make_own_index_fg, with_links=True)

        assert snapshot.escaped is None, f"nothing may cross evaluate: {snapshot.escaped}"
        assert snapshot.identified == (), f"an unlinkable candidate must win nothing, got: {snapshot.identified}"
        assert snapshot.eliminations == ((OWN_INDEX_CLASS_NAME, "links", LINKS_REASON),), (
            f"exactly one links elimination, got: {snapshot.eliminations}"
        )
        assert snapshot.calls >= 1, "evaluate must have read the candidate's index columns"


class TestEnrichmentPrecedence:
    """unify_options enriches the per-match copy before the criteria gate."""

    def test_the_matched_copy_keeps_declared_values_and_gains_missing_keys_by_category(self) -> None:
        snapshot = _drive_enriched_matching()

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.names == (FILTER_FEATURE,), f"the enriched filter must attach, got: {snapshot.names}"
        assert snapshot.matched_group == (
            (GROUP_ONLY_KEY_728, GROUP_ONLY_VALUE),
            (SHARED_KEY_728, FILTER_VALUE),
        ), f"declared values stay, group keys enrich into group, got: {snapshot.matched_group}"
        assert snapshot.matched_context == ((CONTEXT_ONLY_KEY_728, CONTEXT_ONLY_VALUE),), (
            f"context keys must enrich into context, got: {snapshot.matched_context}"
        )

    def test_the_criteria_hook_observes_the_enriched_view(self) -> None:
        """The hook reads the union, with declared values on top."""
        snapshot = _drive_enriched_matching()

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.observed == (
            (
                (SHARED_KEY_728, FILTER_VALUE),
                (GROUP_ONLY_KEY_728, GROUP_ONLY_VALUE),
                (CONTEXT_ONLY_KEY_728, CONTEXT_ONLY_VALUE),
            ),
        ), f"one hook call, seeing the enriched view, got: {snapshot.observed}"

    def test_the_stored_original_filter_is_untouched(self) -> None:
        """The deepcopy is the attachment snapshot; the shared key's divergence WARNING is tolerated by design."""
        snapshot = _drive_enriched_matching()

        assert snapshot.stored_group == ((SHARED_KEY_728, FILTER_VALUE),), (
            f"the stored original must keep only its declared key, got: {snapshot.stored_group}"
        )
        assert snapshot.stored_context == (), f"no enriched key may reach the original, got: {snapshot.stored_context}"

    def test_a_context_declared_key_blocks_the_hosts_group_value(self) -> None:
        """Membership spans both categories: a key declared in context never re-enriches from the host's group."""
        snapshot = _drive_enriched_matching(Options(context={SHARED_KEY_728: FILTER_VALUE}))

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.names == (FILTER_FEATURE,), f"the filter must still attach, got: {snapshot.names}"
        assert snapshot.matched_group == ((GROUP_ONLY_KEY_728, GROUP_ONLY_VALUE),), (
            f"the host's group value must not cross into the declared key, got: {snapshot.matched_group}"
        )
        assert snapshot.matched_context == (
            (CONTEXT_ONLY_KEY_728, CONTEXT_ONLY_VALUE),
            (SHARED_KEY_728, FILTER_VALUE),
        ), f"the declared context value must stay where it was declared, got: {snapshot.matched_context}"

    def test_the_canonical_seam_passes_the_declared_options_unenriched(self) -> None:
        """No host exists on the resolution seam, so the hook sees the declared options and nothing more."""
        snapshot = _drive_canonical_declared_options()

        assert snapshot.escaped is None, f"nothing may cross evaluate: {snapshot.escaped}"
        assert snapshot.observed == (
            (
                (SHARED_KEY_728, FILTER_VALUE),
                (GROUP_ONLY_KEY_728, "None"),
                (CONTEXT_ONLY_KEY_728, "None"),
            ),
        ), f"one hook call, seeing only the declared options, got: {snapshot.observed}"
