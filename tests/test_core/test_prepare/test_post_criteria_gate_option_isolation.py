"""os-061: a candidate whose criteria match SUCCEEDS but is then rejected by a LATER gate in
``_filter_loop`` (domain here) must not leak the options it wrote into a later-visited candidate.
Doubles carry an ``os061`` suffix and are dropped per test.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any
from typing import TypeVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


SHARED_FEATURE = "shared_post_criteria_gate_feat_os061"
LEAK_KEY = "leaked_option_os061"
LEAK_VALUE = "written_by_the_domain_rejected_candidate_os061"
REQUESTED_DOMAIN = "requested_domain_os061"
WRONG_DOMAIN = "wrong_domain_os061"
REJECTED_CLASS_NAME = "DomainRejectedMatchFG_os061"
WINNER_CLASS_NAME = "DomainMatchingFG_os061"

T = TypeVar("T")


class PostCriteriaGateFw_os061(ComputeFramework):
    """Dummy compute framework for the post-criteria-gate option-isolation tests."""


def _capture(call: Callable[[], T]) -> tuple[T | None, str | None]:
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (an escape, or its absence, is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _make_domain_rejected_match_fg() -> type[FeatureGroup]:
    """Criteria match RETURNS TRUE and writes a distinctive option, but the domain gate rejects it."""
    gc.collect()

    class DomainRejectedMatchFG_os061(FeatureGroup):
        """Stands in for a matched candidate that a later gate still eliminates."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {SHARED_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {PostCriteriaGateFw_os061}

        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(WRONG_DOMAIN)

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) != SHARED_FEATURE:
                return False
            options.add_to_group(LEAK_KEY, LEAK_VALUE)
            return True

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return DomainRejectedMatchFG_os061


def _make_domain_matching_fg() -> type[FeatureGroup]:
    """Clean winner: matches the shared feature name and the requested domain, writes nothing."""
    gc.collect()

    class DomainMatchingFG_os061(FeatureGroup):
        """The winner whose options must not carry the eliminated candidate's write."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {SHARED_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {PostCriteriaGateFw_os061}

        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(REQUESTED_DOMAIN)

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return DomainMatchingFG_os061


@dataclass(frozen=True)
class _OptionsSnapshot:
    """Plain-data readout of one evaluation. Holds no class and no exception object."""

    escaped: str | None
    identified_names: tuple[str, ...]
    option_keys: tuple[str, ...]
    leak_value: str | None


def _evaluate(builders: tuple[Callable[[], type[FeatureGroup]], ...]) -> _OptionsSnapshot:
    """Evaluate the shared feature against the given candidates, in order, and read the options out."""
    candidates = [build() for build in builders]
    try:
        feature = Feature(SHARED_FEATURE, domain=REQUESTED_DOMAIN)
        plugins: FeatureGroupEnvironmentMapping = {candidate: {PostCriteriaGateFw_os061} for candidate in candidates}
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None))
        identified = () if result is None else tuple(sorted(fg.get_class_name() for fg in result.identified))
        snapshot = _OptionsSnapshot(
            escaped=escaped,
            identified_names=identified,
            option_keys=tuple(sorted(str(key) for key in feature.options.keys())),
            leak_value=feature.options.get(LEAK_KEY),
        )
        del result
        del plugins
        del feature
        return snapshot
    finally:
        candidates.clear()
        del candidates
        gc.collect()


class TestPostCriteriaGateRejectionLeavesNoPartialMutation:
    """A candidate eliminated AFTER a successful criteria match must not configure the feature that won."""

    def test_write_from_a_domain_rejected_candidate_does_not_leak_into_a_later_candidate(self) -> None:
        snapshot = _evaluate((_make_domain_rejected_match_fg, _make_domain_matching_fg))

        assert snapshot.escaped is None
        assert snapshot.identified_names == (WINNER_CLASS_NAME,)
        assert snapshot.leak_value is None, (
            f"a post-criteria gate rejection must leave no partial mutation, found {LEAK_KEY}={snapshot.leak_value}"
        )
        assert LEAK_KEY not in snapshot.option_keys


class TestPostCriteriaGateLeakIsVisitOrderDependent:
    """Visiting the winner first leaves nothing for the rejected candidate to leak into."""

    def test_earlier_winner_is_unaffected_by_a_later_domain_rejected_candidate(self) -> None:
        snapshot = _evaluate((_make_domain_matching_fg, _make_domain_rejected_match_fg))

        assert snapshot.escaped is None
        assert snapshot.identified_names == (WINNER_CLASS_NAME,)
        assert snapshot.leak_value is None
        assert LEAK_KEY not in snapshot.option_keys


ORDER_KEY = "order_shared_key_os062"
ORIGINAL_KEY = "original_key_os062"


class ReaderParent_os062(BaseInputData):
    """Reader class of the parent candidate."""


class ReaderSub_os062(BaseInputData):
    """Reader class of the subclass candidate."""


@dataclass(frozen=True)
class _Writes:
    group: dict[str, Any]
    context: dict[str, Any]
    non_forwarded: frozenset[str]
    reader: tuple[type[BaseInputData], str] | None = None


def _make_candidate(
    name: str, writes: _Writes, seen: list[dict[str, Any]], base: type[FeatureGroup] = FeatureGroup
) -> type[FeatureGroup]:
    """Build a candidate that records the group options it was shown, then applies ``writes`` and matches."""

    def match(
        cls: type[FeatureGroup],
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) != SHARED_FEATURE:
            return False
        seen.append(dict(options.group))
        options.group.update(writes.group)
        options.context.update(writes.context)
        options.non_forwarded_group_keys = options.non_forwarded_group_keys | writes.non_forwarded
        if writes.reader is not None:
            BaseInputData.add_base_input_data_to_options(writes.reader[0], writes.reader[1], options)
        return True

    def names(cls: type[FeatureGroup]) -> set[str]:
        return {SHARED_FEATURE}

    def frameworks(cls: type[FeatureGroup]) -> set[type[ComputeFramework]] | None:
        return {PostCriteriaGateFw_os061}

    def inputs(self: FeatureGroup, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    namespace = {
        "match_feature_group_criteria": classmethod(match),
        "feature_names_supported": classmethod(names),
        "compute_framework_rule": classmethod(frameworks),
        "input_features": inputs,
    }
    return type(name, (base,), namespace)


@dataclass(frozen=True)
class _ResolvedOptions:
    escaped: str | None
    winners: tuple[str, ...]
    group: dict[str, Any]
    context: dict[str, Any]
    non_forwarded: frozenset[str]


def _resolve(candidates: list[type[FeatureGroup]]) -> _ResolvedOptions:
    feature = Feature(SHARED_FEATURE, options={ORIGINAL_KEY: "original"})
    plugins: FeatureGroupEnvironmentMapping = {c: {PostCriteriaGateFw_os061} for c in candidates}
    result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None))
    winners = () if result is None else tuple(sorted(fg.get_class_name() for fg in result.identified))
    resolved = _ResolvedOptions(
        escaped=escaped,
        winners=winners,
        group=dict(feature.options.group),
        context=dict(feature.options.context),
        non_forwarded=feature.options.non_forwarded_group_keys,
    )
    del result
    gc.collect()
    return resolved


def _parent_and_sub(
    seen: list[dict[str, Any]], with_reader: bool = False
) -> tuple[type[FeatureGroup], type[FeatureGroup]]:
    gc.collect()
    parent_writes = _Writes(
        group={ORDER_KEY: "parent", "parent_only_os062": "p"},
        context={"parent_ctx_only_os062": "p", "ctx_shared_os062": "parent"},
        non_forwarded=frozenset({"parent_nf_os062"}),
        reader=(ReaderParent_os062, "parent_access") if with_reader else None,
    )
    sub_writes = _Writes(
        group={ORDER_KEY: "sub"},
        context={"ctx_shared_os062": "sub"},
        non_forwarded=frozenset({"sub_nf_os062"}),
        reader=(ReaderSub_os062, "sub_access") if with_reader else None,
    )
    parent = _make_candidate("ParentFG_os062", parent_writes, seen)
    sub = _make_candidate("SubFG_os062", sub_writes, seen, base=parent)
    return parent, sub


@pytest.mark.parametrize("parent_first", [True, False])
class TestWinnerOptionsAreOriginalPlusOwnWrites:
    """A dropped parent's matcher writes must not survive on the feature the winning subclass computes."""

    def test_options_hold_original_plus_winner_writes_only(self, parent_first: bool) -> None:
        parent, sub = _parent_and_sub([])
        resolved = _resolve([parent, sub] if parent_first else [sub, parent])

        assert resolved.escaped is None
        assert resolved.winners == ("SubFG_os062",)
        assert resolved.group == {ORIGINAL_KEY: "original", ORDER_KEY: "sub"}
        assert resolved.context == {"ctx_shared_os062": "sub"}
        assert resolved.non_forwarded == frozenset({"sub_nf_os062"})

    def test_different_reader_pairs_resolve_to_the_subclass_pair(self, parent_first: bool) -> None:
        parent, sub = _parent_and_sub([], with_reader=True)
        resolved = _resolve([parent, sub] if parent_first else [sub, parent])

        assert resolved.escaped is None
        assert resolved.winners == ("SubFG_os062",)
        assert resolved.group[BaseInputData.__name__] == (ReaderSub_os062, "sub_access")


class TestEachCandidateSeesTheOriginalOptions:
    """A candidate's matcher must see the request's options, not a sibling candidate's write."""

    @pytest.mark.parametrize("first_wins_order", [True, False])
    def test_unrelated_candidates_do_not_see_each_others_writes(self, first_wins_order: bool) -> None:
        gc.collect()
        seen: list[dict[str, Any]] = []
        one = _make_candidate("UnrelatedOneFG_os062", _Writes({"one_key_os062": 1}, {}, frozenset()), seen)
        two = _make_candidate("UnrelatedTwoFG_os062", _Writes({"two_key_os062": 2}, {}, frozenset()), seen)
        resolved = _resolve([one, two] if first_wins_order else [two, one])

        assert resolved.escaped is None
        assert len(seen) == 2
        assert seen == [{ORIGINAL_KEY: "original"}, {ORIGINAL_KEY: "original"}]


class TestUnrelatedDifferentReadersStillConflict:
    """Two unrelated survivors with different reader pairs keep raising the double-reader error."""

    @pytest.mark.parametrize("one_first", [True, False])
    def test_unrelated_candidates_with_different_reader_pairs_raise(self, one_first: bool) -> None:
        gc.collect()
        one = _make_candidate("ReaderOneFG_os062", _Writes({}, {}, frozenset(), (ReaderParent_os062, "one_access")), [])
        two = _make_candidate("ReaderTwoFG_os062", _Writes({}, {}, frozenset(), (ReaderSub_os062, "two_access")), [])
        resolved = _resolve([one, two] if one_first else [two, one])

        assert resolved.escaped is not None
        assert "BaseInputData already set with different values" in resolved.escaped
