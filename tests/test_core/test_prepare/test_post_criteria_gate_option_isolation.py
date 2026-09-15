"""os-061: a candidate whose criteria match SUCCEEDS but is then rejected by a LATER gate in
``_filter_loop`` (domain here) must not leak the options it wrote into a later-visited candidate.
Doubles carry an ``os061`` suffix and are dropped per test.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TypeVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
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
