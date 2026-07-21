"""Pinning tests for match-time enforcement of SubtypeDeclaration (issue #639)."""

import re
from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import CandidateFrameworks, IdentifyFeatureGroupClass
from mloda.provider import FeatureChainParserMixin, FeatureGroup, SubtypeDeclaration, property_spec
from tests.test_core.test_prepare.identify_seam import identify_winner


MATCH_KEY = "subdeclm_window_function"
MATCH_WINDOW_UNIVERSE = frozenset({"median", "sum", "lag", "ntile"})
MATCH_BETA_SUPPORTED = frozenset({"sum", "lag"})
MATCH_PLAIN_FEATURE = "subdeclm_plain_feature"
COMPILED_KEY = "subdeclmc_window_function"
COMPILED_BETA_SUPPORTED = frozenset({"sum", "lag"})
COMPILED_SUFFIX_KEY = "subdeclmcs_window_function"
COMPILED_SUFFIX_BETA_SUPPORTED = frozenset({"sum", "lag"})
RANK_UNIVERSE = frozenset({"dense", "ordinal", "ntile"})
FRAME_LITERALS = frozenset({"rows_1", "rows_7"})
FRAME_UNIVERSE = FRAME_LITERALS | {"rows"}


class SubDeclMatchFwAlpha(ComputeFramework):
    """First dummy compute framework for subtype-matching tests."""


class SubDeclMatchFwBeta(ComputeFramework):
    """Second dummy compute framework for subtype-matching tests."""


class SubDeclMatchWindowFG(FeatureChainParserMixin, FeatureGroup):
    """Keyed declaration narrowing subtype support on SubDeclMatchFwBeta."""

    SUBTYPES = SubtypeDeclaration(
        key=MATCH_KEY,
        parametric_families={"ntile": "N-tile bucketing"},
        supported={SubDeclMatchFwBeta.get_class_name(): MATCH_BETA_SUPPORTED},
    )
    PREFIX_PATTERN = r".*__([\w]+)_subdeclmw$"
    PROPERTY_MAPPING = {
        MATCH_KEY: property_spec(
            "Window function subtype.",
            strict=True,
            allowed_values={"median": "Median", "sum": "Sum", "lag": "Lag"},
            deferred_binding=True,  # parametric subtypes (ntile_2) are resolved by SUBTYPES, not name capture
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}


class SubDeclMatchCompiledWindowFG(FeatureChainParserMixin, FeatureGroup):
    """Compiled-pattern twin of SubDeclMatchWindowFG; a compiled PREFIX_PATTERN must resolve like a str one."""

    SUBTYPES = SubtypeDeclaration(
        key=COMPILED_KEY,
        parametric_families={"ntile": "N-tile bucketing"},
        supported={SubDeclMatchFwBeta.get_class_name(): COMPILED_BETA_SUPPORTED},
    )
    PREFIX_PATTERN = re.compile(r".*__([\w]+)_subdeclmcompiled$")
    PROPERTY_MAPPING = {
        COMPILED_KEY: property_spec(
            "Window function subtype.",
            strict=True,
            allowed_values={"median": "Median", "sum": "Sum", "lag": "Lag"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}


class SubDeclMatchCompiledSuffixFG(FeatureChainParserMixin, FeatureGroup):
    """Compiled-SUFFIX_PATTERN twin: prefix_patterns_of gathers SUFFIX_PATTERN too, so it must resolve alike."""

    SUBTYPES = SubtypeDeclaration(
        key=COMPILED_SUFFIX_KEY,
        parametric_families={"ntile": "N-tile bucketing"},
        supported={SubDeclMatchFwBeta.get_class_name(): COMPILED_SUFFIX_BETA_SUPPORTED},
    )
    SUFFIX_PATTERN = re.compile(r".*__([\w]+)_subdeclmsuffix$")
    PROPERTY_MAPPING = {
        COMPILED_SUFFIX_KEY: property_spec(
            "Window function subtype.",
            strict=True,
            allowed_values={"median": "Median", "sum": "Sum", "lag": "Lag"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}


class SubDeclMatchPlainFG(FeatureGroup):
    """Family without any subtype dimension."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == MATCH_PLAIN_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SubDeclMatchRankLikeFG(FeatureChainParserMixin, FeatureGroup):
    """Registry-like keyed parametric family: rank types plus N-tile bucketing."""

    SUBTYPES = SubtypeDeclaration(
        key="rank_type",
        parametric_families={"ntile": "N-tile bucketing"},
        supported={SubDeclMatchFwBeta.get_class_name(): {"dense"}},
    )
    PREFIX_PATTERN = r".*__([\w]+)_subdeclmrank$"
    PROPERTY_MAPPING = {
        "rank_type": property_spec(
            "Ranking method.",
            strict=True,
            allowed_values={"dense": "Dense rank", "ordinal": "Ordinal rank"},
            deferred_binding=True,  # parametric subtypes (ntile_2) are resolved by SUBTYPES, not name capture
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}


def _subdeclm_frame_resolver(feature_name: str, options: Options) -> Optional[str]:
    match = re.match(r".*__(rows_\d+)_frame$", feature_name)
    if match is None:
        return None
    return match.group(1)


class SubDeclMatchFrameSpecFG(FeatureGroup):
    """Registry-like shape B family: flattened frame spec with a name resolver."""

    SUBTYPES = SubtypeDeclaration(
        universe=FRAME_LITERALS,
        resolver=_subdeclm_frame_resolver,
        parametric_families={"rows": "Row-count frames"},
        supported={SubDeclMatchFwBeta.get_class_name(): {"rows_1"}},
    )

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name).startswith("subdeclm_") and str(feature_name).endswith("_frame")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def _identify(
    feature: Feature, accessible_plugins: FeatureGroupEnvironmentMapping
) -> tuple[type[FeatureGroup], set[type[ComputeFramework]]]:
    return identify_winner(feature, accessible_plugins)


class TestMatchTimeIntegration:
    """The declaration is enforced through IdentifyFeatureGroupClass."""

    def test_unsupported_subtype_is_routed_to_capable_framework(self) -> None:
        feature = Feature("value__median_subdeclmw")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchWindowFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        feature_group_class, compute_frameworks = _identify(feature, accessible_plugins)
        assert feature_group_class is SubDeclMatchWindowFG
        assert compute_frameworks == {SubDeclMatchFwAlpha}, (
            f"'median' is unsupported on SubDeclMatchFwBeta and must be routed around; got {compute_frameworks}"
        )

    def test_pin_to_incapable_framework_raises_capability_error(self) -> None:
        feature = Feature("value__median_subdeclmw")
        feature.compute_frameworks = {SubDeclMatchFwBeta}
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchWindowFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        with pytest.raises(ValueError) as exc_info:
            _identify(feature, accessible_plugins)

        message = str(exc_info.value)
        assert "compute framework pin" in message, (
            f"Error must signal a rejected compute-framework pin, but got: {message}"
        )
        assert "is not among its supported" in message, (
            f"Error must signal an unsupported framework, but got: {message}"
        )
        assert SubDeclMatchFwBeta.get_class_name() in message
        assert SubDeclMatchFwAlpha.get_class_name() in message
        assert "Did you mean" not in message

    def test_parametric_instance_is_routed_around(self) -> None:
        feature = Feature("value__ntile_2_subdeclmw")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchWindowFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        feature_group_class, compute_frameworks = _identify(feature, accessible_plugins)
        assert feature_group_class is SubDeclMatchWindowFG
        assert compute_frameworks == {SubDeclMatchFwAlpha}, (
            f"Family 'ntile' is unsupported on SubDeclMatchFwBeta, so 'ntile_2' must be routed around; "
            f"got {compute_frameworks}"
        )

    def test_unknown_subtype_keeps_every_framework(self) -> None:
        assert SubDeclMatchWindowFG.canonical_subtype("zzz") == "zzz"

        feature = Feature("value__zzz_subdeclmw")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchWindowFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        _, compute_frameworks = _identify(feature, accessible_plugins)
        assert compute_frameworks == {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}, (
            f"An undeclared subtype must stay open on every framework; got {compute_frameworks}"
        )

    def test_family_without_subtype_dimension_is_unaffected(self) -> None:
        assert SubDeclMatchPlainFG.SUBTYPES is None
        assert SubDeclMatchPlainFG.subtype_universe() == frozenset()

        feature = Feature(MATCH_PLAIN_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchPlainFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        _, compute_frameworks = _identify(feature, accessible_plugins)
        assert compute_frameworks == {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}

    def test_matching_stays_orthogonal_to_subtype_support(self) -> None:
        assert SubDeclMatchWindowFG.resolve_subtype("value__median_subdeclmw", Options()) == "median"
        assert "median" not in SubDeclMatchWindowFG.supported_subtypes(SubDeclMatchFwBeta)
        assert SubDeclMatchWindowFG.match_feature_group_criteria("value__median_subdeclmw", Options()) is True


class TestRankLikeFamily:
    """Definition of done: a keyed parametric registry-like family routes by declaration alone."""

    def test_universe_enumerates_family_without_probing(self) -> None:
        assert SubDeclMatchRankLikeFG.subtype_universe() == RANK_UNIVERSE

    def test_parametric_instance_canonicalizes_to_family(self) -> None:
        assert SubDeclMatchRankLikeFG.canonical_subtype("ntile_2") == "ntile"

    def test_parametric_instance_routes_only_to_capable_framework(self) -> None:
        feature = Feature("value__ntile_2_subdeclmrank")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchRankLikeFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        feature_group_class, compute_frameworks = _identify(feature, accessible_plugins)
        assert feature_group_class is SubDeclMatchRankLikeFG
        assert compute_frameworks == {SubDeclMatchFwAlpha}, (
            f"'ntile' is only supported on SubDeclMatchFwAlpha; got {compute_frameworks}"
        )

    def test_supported_subtype_runs_on_both_frameworks(self) -> None:
        feature = Feature("value__dense_subdeclmrank")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchRankLikeFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        _, compute_frameworks = _identify(feature, accessible_plugins)
        assert compute_frameworks == {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}


class TestFrameSpecLikeFamily:
    """Definition of done: a shape B registry-like family resolves from names and routes by narrowing."""

    def test_universe_unions_literals_and_family(self) -> None:
        assert SubDeclMatchFrameSpecFG.subtype_universe() == FRAME_UNIVERSE

    def test_resolver_parses_subtype_from_name(self) -> None:
        assert SubDeclMatchFrameSpecFG.resolve_subtype("subdeclm_price__rows_7_frame", Options()) == "rows_7"

    def test_declared_literal_never_collapses_parametrically(self) -> None:
        # 'rows_7' is a declared literal; 'rows' looking like a family stem must not swallow it.
        assert SubDeclMatchFrameSpecFG.canonical_subtype("rows_7") == "rows_7"
        assert SubDeclMatchFrameSpecFG.canonical_subtype("rows_9") == "rows"

    def test_routing_honors_supported_narrowing(self) -> None:
        feature = Feature("subdeclm_price__rows_7_frame")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchFrameSpecFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        feature_group_class, compute_frameworks = _identify(feature, accessible_plugins)
        assert feature_group_class is SubDeclMatchFrameSpecFG
        assert compute_frameworks == {SubDeclMatchFwAlpha}, (
            f"'rows_7' is unsupported on SubDeclMatchFwBeta; got {compute_frameworks}"
        )

    def test_supported_literal_runs_on_both_frameworks(self) -> None:
        feature = Feature("subdeclm_price__rows_1_frame")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchFrameSpecFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        _, compute_frameworks = _identify(feature, accessible_plugins)
        assert compute_frameworks == {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}


def _subdeclm_raising_resolver(feature_name: str, options: Options) -> Optional[str]:
    raise RuntimeError("subdeclm resolver boom")


class SubDeclMatchRaisingResolverFG(FeatureGroup):
    """Shape B family whose resolver raises; planning must degrade open."""

    SUBTYPES = SubtypeDeclaration(
        universe={"boom"},
        resolver=_subdeclm_raising_resolver,
        supported={SubDeclMatchFwBeta.get_class_name(): frozenset()},
    )

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestSubDeclMatchPlanningNeverRaises:
    """A raising resolver must not crash the planning path; the family degrades open."""

    def test_raising_resolver_degrades_open_in_split(self) -> None:
        """Both frameworks stay supported: the split the evaluation seam records keeps neither rejected."""
        feature = Feature(SubDeclMatchRaisingResolverFG.get_class_name())
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchRaisingResolverFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        result = IdentifyFeatureGroupClass.evaluate(feature=feature, accessible_plugins=accessible_plugins, links=None)

        assert result.candidate_frameworks[SubDeclMatchRaisingResolverFG] == CandidateFrameworks(
            supported=frozenset({SubDeclMatchFwAlpha, SubDeclMatchFwBeta}),
            rejected=frozenset(),
        )


class TestCompiledPrefixPatternParity:
    """A compiled PREFIX_PATTERN must resolve and route exactly like its string sibling (issue #765)."""

    def test_compiled_pattern_resolves_same_subtype_as_string_sibling(self) -> None:
        # The matcher accepts the feature on both forms; resolution must not fail open on the compiled one.
        assert SubDeclMatchCompiledWindowFG.match_feature_group_criteria("value__median_subdeclmcompiled", Options())
        string_subtype = SubDeclMatchWindowFG.resolve_subtype("value__median_subdeclmw", Options())
        compiled_subtype = SubDeclMatchCompiledWindowFG.resolve_subtype("value__median_subdeclmcompiled", Options())
        assert string_subtype == "median"
        assert compiled_subtype == string_subtype, (
            f"A compiled PREFIX_PATTERN must resolve the subtype like the string form '{string_subtype}'; "
            f"the guard drops the re.Pattern and returns {compiled_subtype}."
        )

    def test_unsupported_subtype_routes_around_incapable_framework(self) -> None:
        feature = Feature("value__median_subdeclmcompiled")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchCompiledWindowFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        feature_group_class, compute_frameworks = _identify(feature, accessible_plugins)
        assert feature_group_class is SubDeclMatchCompiledWindowFG
        assert compute_frameworks == {SubDeclMatchFwAlpha}, (
            f"'median' is unsupported on SubDeclMatchFwBeta and must be routed around even when PREFIX_PATTERN "
            f"is compiled; the dropped pattern fails open and keeps {compute_frameworks}"
        )

    def test_supported_subtype_runs_on_both_frameworks(self) -> None:
        feature = Feature("value__sum_subdeclmcompiled")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchCompiledWindowFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        _, compute_frameworks = _identify(feature, accessible_plugins)
        # Pin resolution directly: pre-fix the dropped compiled pattern returns None, so routing merely fails open.
        assert SubDeclMatchCompiledWindowFG.resolve_subtype("value__sum_subdeclmcompiled", Options()) == "sum"
        assert compute_frameworks == {SubDeclMatchFwAlpha, SubDeclMatchFwBeta}, (
            f"'sum' is supported on SubDeclMatchFwBeta and must run on both frameworks; got {compute_frameworks}"
        )

    def test_compiled_suffix_pattern_resolves_and_routes_like_prefix(self) -> None:
        # resolve_subtype gathers SUFFIX_PATTERN via prefix_patterns_of, so a compiled suffix must resolve alike.
        assert SubDeclMatchCompiledSuffixFG.resolve_subtype("value__median_subdeclmsuffix", Options()) == "median"

        feature = Feature("value__median_subdeclmsuffix")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubDeclMatchCompiledSuffixFG: {SubDeclMatchFwAlpha, SubDeclMatchFwBeta},
        }

        feature_group_class, compute_frameworks = _identify(feature, accessible_plugins)
        assert feature_group_class is SubDeclMatchCompiledSuffixFG
        assert compute_frameworks == {SubDeclMatchFwAlpha}, (
            f"'median' is unsupported on SubDeclMatchFwBeta and must be routed around even when the compiled "
            f"pattern is a SUFFIX_PATTERN; the dropped pattern fails open and keeps {compute_frameworks}"
        )
