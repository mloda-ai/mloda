"""Paired engine/diagnostic characterization tests for issue #722 Stage 1.

Every test PASSES against current code: each pair pins one divergence between the
engine resolution path (IdentifyFeatureGroupClass) and the debug resolution path
(resolve_feature). Assertions marked "PINS CURRENT DIVERGENCE" pin behavior that
Stage 3/4 will change; assertions marked "TARGET CONTRACT" pin behavior to keep.

Index (divergence -> engine test / resolve_feature test):
- #5 domain:
    test_engine_domain_disambiguates_shared_name
    test_resolve_feature_has_no_domain_parameter_reports_false_conflict
- #6 links/index:
    test_engine_links_filter_excludes_unsupported_index_probe
    test_resolve_feature_has_no_links_parameter_resolves_phantom
- #7 compute-framework pin:
    test_engine_compute_framework_pin_disambiguates_siblings
    test_resolve_feature_has_no_framework_pin_reports_conflict
- #10 data access collection:
    test_engine_data_access_collection_enables_reader_probe
    test_resolve_feature_hardcodes_none_data_access_collection
- #11 ValueError from matching:
    test_engine_matching_value_error_propagates_despite_clean_winner
    test_resolve_feature_hides_matching_value_error_behind_clean_winner
- #12 option-aware hints:
    test_engine_no_match_error_carries_extra_group_option_hint
    test_resolve_feature_no_match_error_lacks_group_option_hint
- #14 multiple-match text:
    test_engine_multiple_match_error_carries_module_paths_and_url
    test_resolve_feature_multiple_match_error_bare_names_only
- #15 not-found text:
    test_engine_not_found_error_carries_suggestions_and_url
    test_resolve_feature_not_found_error_is_single_bare_line
"""

from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.user import PluginCollector


TROUBLESHOOTING_URL = "https://mloda-ai.github.io/mloda/in_depth/troubleshooting/feature-group-resolution-errors/"
MODULE_PATH = "tests.test_core.test_resolution_parity.test_parity_request_context"

DOMAIN_FEATURE = "probe722b_domain"
DOMAIN_A = "probe722b_domain_a"
DOMAIN_B = "probe722b_domain_b"
INDEXED_FEATURE = "probe722b_indexed"
PIN_FEATURE = "probe722b_pin"
DAC_FEATURE = "probe722b_dac"
BOOM_FEATURE = "probe722b_boom"
PICKY_FEATURE = "probe722b_picky"
MULTI_FEATURE = "probe722b_multi"
NEEDLE_FEATURE = "probe722b_needle"
NEEDLE_TYPO = "probe722b_needls"


class CfwParity722B(ComputeFramework):
    """Framework for hand-built engine mappings in this module."""


class CfwPinX722B(ComputeFramework):
    """Uniquely named framework, pinnable by name from a Feature."""


class CfwPinY722B(ComputeFramework):
    """Rival uniquely named framework for the sibling probe."""


class _ParityProbeBase722B(FeatureGroup):
    """Shared probe base: never matches anything itself."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


# ---------------------------------------------------------------------------
# Divergence 5: domain
# ---------------------------------------------------------------------------


class ProbeDomainA722B(_ParityProbeBase722B):
    """Matches the shared domain feature name; lives in domain A."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain(DOMAIN_A)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == DOMAIN_FEATURE


class ProbeDomainB722B(_ParityProbeBase722B):
    """Matches the shared domain feature name; lives in domain B."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain(DOMAIN_B)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == DOMAIN_FEATURE


def _domain_pair() -> FeatureGroupEnvironmentMapping:
    return {
        ProbeDomainA722B: {CfwParity722B},
        ProbeDomainB722B: {CfwParity722B},
    }


def test_engine_domain_disambiguates_shared_name() -> None:
    """The engine's domain filter resolves a name two groups share."""
    # Premise guard: without a domain both probes match.
    with pytest.raises(ValueError, match="Multiple feature groups found"):
        IdentifyFeatureGroupClass(
            feature=Feature(DOMAIN_FEATURE),
            accessible_plugins=_domain_pair(),
            links=None,
        )

    identifier = IdentifyFeatureGroupClass(
        feature=Feature(DOMAIN_FEATURE, domain=DOMAIN_A),
        accessible_plugins=_domain_pair(),
        links=None,
    )
    resolved, _ = identifier.get()
    # TARGET CONTRACT
    assert resolved is ProbeDomainA722B


def test_resolve_feature_has_no_domain_parameter_reports_false_conflict() -> None:
    """resolve_feature cannot express the run's domain, so it reports a conflict."""
    collector = PluginCollector.enabled_feature_groups({ProbeDomainA722B, ProbeDomainB722B})

    result = resolve_feature(DOMAIN_FEATURE, plugin_collector=collector)

    # PINS CURRENT DIVERGENCE (#5): debug reports a conflict the run does not have.
    assert result.feature_group is None
    assert result.error is not None
    assert "Multiple FeatureGroups match" in result.error
    assert "ProbeDomainA722B" in result.error
    assert "ProbeDomainB722B" in result.error


# ---------------------------------------------------------------------------
# Divergence 6: links/index
# ---------------------------------------------------------------------------


class ProbeLinkLeft722B(_ParityProbeBase722B):
    """Anchors the left side of the foreign link; never matches a name."""


class ProbeLinkRight722B(_ParityProbeBase722B):
    """Anchors the right side of the foreign link; never matches a name."""


class ProbeIndexed722B(_ParityProbeBase722B):
    """Declares an index no link in the run carries."""

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("probe722b_row_id",))]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == INDEXED_FEATURE


def _foreign_link() -> Link:
    """A link between two other groups whose indexes the indexed probe does not support."""
    return Link.inner(
        JoinSpec(ProbeLinkLeft722B, "probe722b_left_id"),
        JoinSpec(ProbeLinkRight722B, "probe722b_right_id"),
    )


def test_engine_links_filter_excludes_unsupported_index_probe() -> None:
    """The engine filters out an indexed group when no run link carries its index."""
    # Premise guard: without links the probe resolves.
    identifier = IdentifyFeatureGroupClass(
        feature=Feature(INDEXED_FEATURE),
        accessible_plugins={ProbeIndexed722B: {CfwParity722B}},
        links=None,
    )
    resolved, _ = identifier.get()
    assert resolved is ProbeIndexed722B

    # TARGET CONTRACT: the run's links filter eliminates the probe.
    with pytest.raises(ValueError, match="No feature groups found"):
        IdentifyFeatureGroupClass(
            feature=Feature(INDEXED_FEATURE),
            accessible_plugins={ProbeIndexed722B: {CfwParity722B}},
            links={_foreign_link()},
        )


def test_resolve_feature_has_no_links_parameter_resolves_phantom() -> None:
    """resolve_feature cannot express the run's links, so the probe resolves cleanly."""
    collector = PluginCollector.enabled_feature_groups({ProbeIndexed722B})

    result = resolve_feature(INDEXED_FEATURE, plugin_collector=collector)

    # PINS CURRENT DIVERGENCE (#6): phantom resolution; the links filter is unexpressible here.
    assert result.feature_group is ProbeIndexed722B
    assert result.error is None


# ---------------------------------------------------------------------------
# Divergence 7: compute-framework pin
# ---------------------------------------------------------------------------


class ProbePinX722B(_ParityProbeBase722B):
    """Sibling probe on CfwPinX722B, matching the shared pin feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwPinX722B}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == PIN_FEATURE


class ProbePinY722B(_ParityProbeBase722B):
    """Sibling probe on CfwPinY722B, matching the same pin feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwPinY722B}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == PIN_FEATURE


def _pin_pair() -> FeatureGroupEnvironmentMapping:
    return {
        ProbePinX722B: {CfwPinX722B},
        ProbePinY722B: {CfwPinY722B},
    }


def test_engine_compute_framework_pin_disambiguates_siblings() -> None:
    """A framework pin on the Feature resolves the per-framework sibling ambiguity."""
    # Premise guard: unpinned, the siblings are ambiguous.
    with pytest.raises(ValueError, match="Multiple feature groups found"):
        IdentifyFeatureGroupClass(
            feature=Feature(PIN_FEATURE),
            accessible_plugins=_pin_pair(),
            links=None,
        )

    identifier = IdentifyFeatureGroupClass(
        feature=Feature(PIN_FEATURE, compute_framework=CfwPinX722B.get_class_name()),
        accessible_plugins=_pin_pair(),
        links=None,
    )
    resolved, compute_frameworks = identifier.get()
    # TARGET CONTRACT
    assert resolved is ProbePinX722B
    assert compute_frameworks == {CfwPinX722B}


def test_resolve_feature_has_no_framework_pin_reports_conflict() -> None:
    """resolve_feature cannot express a framework pin, so the siblings stay ambiguous."""
    collector = PluginCollector.enabled_feature_groups({ProbePinX722B, ProbePinY722B})

    result = resolve_feature(PIN_FEATURE, plugin_collector=collector)

    # PINS CURRENT DIVERGENCE (#7): the documented remedy (a framework pin) is unexpressible here.
    assert result.feature_group is None
    assert result.error is not None
    assert "Multiple FeatureGroups match" in result.error
    assert "ProbePinX722B" in result.error
    assert "ProbePinY722B" in result.error


# ---------------------------------------------------------------------------
# Divergence 10: data access collection
# ---------------------------------------------------------------------------


class ProbeDac722B(_ParityProbeBase722B):
    """Matches its name only when a data access collection is present."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == DAC_FEATURE and data_access_collection is not None


def test_engine_data_access_collection_enables_reader_probe() -> None:
    """The engine threads the run's data access collection into matching."""
    # Premise guard: without a collection the probe does not match.
    with pytest.raises(ValueError, match="No feature groups found"):
        IdentifyFeatureGroupClass(
            feature=Feature(DAC_FEATURE),
            accessible_plugins={ProbeDac722B: {CfwParity722B}},
            links=None,
            data_access_collection=None,
        )

    identifier = IdentifyFeatureGroupClass(
        feature=Feature(DAC_FEATURE),
        accessible_plugins={ProbeDac722B: {CfwParity722B}},
        links=None,
        data_access_collection=DataAccessCollection(files={"probe722b": "probe722b.csv"}),
    )
    resolved, _ = identifier.get()
    # TARGET CONTRACT
    assert resolved is ProbeDac722B


def test_resolve_feature_hardcodes_none_data_access_collection() -> None:
    """resolve_feature always passes data_access_collection=None into matching."""
    collector = PluginCollector.enabled_feature_groups({ProbeDac722B})

    result = resolve_feature(DAC_FEATURE, plugin_collector=collector)

    # PINS CURRENT DIVERGENCE (#10): reader/input_data groups fail to resolve in the debug tool.
    assert result.feature_group is None
    assert result.error is not None
    assert "No FeatureGroup found" in result.error


# ---------------------------------------------------------------------------
# Divergence 11: ValueError raised by matching
# ---------------------------------------------------------------------------


class ProbeBoom722B(_ParityProbeBase722B):
    """Raises ValueError from matching, gated on its unique shared name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if str(feature_name) == BOOM_FEATURE:
            raise ValueError("boom 722b: matching rejected the request")
        return False


class ProbeWinner722B(_ParityProbeBase722B):
    """Matches the boom feature name cleanly."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == BOOM_FEATURE


def test_engine_matching_value_error_propagates_despite_clean_winner() -> None:
    """A plain ValueError from any candidate's matching aborts the whole engine run."""
    # The winner is inserted first; the raising candidate later in the loop still aborts.
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ProbeWinner722B: {CfwParity722B},
        ProbeBoom722B: {CfwParity722B},
    }

    # Premise guard: on its own the winner probe resolves cleanly, so the ValueError overrides a clean winner.
    identifier = IdentifyFeatureGroupClass(
        feature=Feature(BOOM_FEATURE),
        accessible_plugins={ProbeWinner722B: {CfwParity722B}},
        links=None,
    )
    assert identifier.get()[0] is ProbeWinner722B

    # PINS CURRENT DIVERGENCE (#11): the run aborts even though another candidate matched.
    with pytest.raises(ValueError, match="boom 722b"):
        IdentifyFeatureGroupClass(
            feature=Feature(BOOM_FEATURE),
            accessible_plugins=accessible_plugins,
            links=None,
        )


def test_resolve_feature_hides_matching_value_error_behind_clean_winner() -> None:
    """resolve_feature degrades the raising candidate to a non-match and crowns the winner."""
    collector = PluginCollector.enabled_feature_groups({ProbeBoom722B, ProbeWinner722B})

    result = resolve_feature(BOOM_FEATURE, plugin_collector=collector)

    # PINS CURRENT DIVERGENCE (#11): debug reports a clean winner while the engine crashes.
    assert result.feature_group is ProbeWinner722B
    assert result.error is None
    assert ProbeBoom722B not in result.candidates


# ---------------------------------------------------------------------------
# Divergence 12: option-aware hints
# ---------------------------------------------------------------------------


class ProbePicky722B(_ParityProbeBase722B):
    """Accepts its name only with no group options; never raises."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == PICKY_FEATURE and not options.group


def test_engine_no_match_error_carries_extra_group_option_hint() -> None:
    """The engine's no-match error names the group options that broke the match."""
    feature = Feature(PICKY_FEATURE, Options(group={"junk722b": 1}))

    with pytest.raises(ValueError, match="No feature groups found") as exc_info:
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins={ProbePicky722B: {CfwParity722B}},
            links=None,
        )

    message = str(exc_info.value)
    # TARGET CONTRACT: the forwarding hint fires for directly-set keys too.
    assert "extra group option(s)" in message
    assert "junk722b" in message
    assert "ProbePicky722B" in message


def test_resolve_feature_no_match_error_lacks_group_option_hint() -> None:
    """resolve_feature reports only the bare no-match line for the same request."""
    collector = PluginCollector.enabled_feature_groups({ProbePicky722B})

    result = resolve_feature(PICKY_FEATURE, options=Options(group={"junk722b": 1}), plugin_collector=collector)

    assert result.feature_group is None
    # PINS CURRENT DIVERGENCE (#12): the engine's option hints cannot be reproduced by the debug tool.
    assert result.error == f"No FeatureGroup found for feature name: {PICKY_FEATURE}"
    assert "extra group option(s)" not in result.error


# ---------------------------------------------------------------------------
# Divergence 14: multiple-match text
# ---------------------------------------------------------------------------


class ProbeMultiA722B(_ParityProbeBase722B):
    """First of two unrelated probes matching the multi feature name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == MULTI_FEATURE


class ProbeMultiB722B(_ParityProbeBase722B):
    """Second of two unrelated probes matching the multi feature name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == MULTI_FEATURE


def test_engine_multiple_match_error_carries_module_paths_and_url() -> None:
    """The engine's ambiguity error carries module paths and ends with the troubleshooting URL."""
    with pytest.raises(ValueError, match="Multiple feature groups found") as exc_info:
        IdentifyFeatureGroupClass(
            feature=Feature(MULTI_FEATURE),
            accessible_plugins={ProbeMultiA722B: {CfwParity722B}, ProbeMultiB722B: {CfwParity722B}},
            links=None,
        )

    message = str(exc_info.value)
    # TARGET CONTRACT: formatted candidates with module paths, URL last.
    assert f"ProbeMultiA722B ({MODULE_PATH})" in message
    assert f"ProbeMultiB722B ({MODULE_PATH})" in message
    assert message.endswith(TROUBLESHOOTING_URL)


def test_resolve_feature_multiple_match_error_bare_names_only() -> None:
    """resolve_feature's ambiguity error carries bare class names, no module paths, no URL."""
    collector = PluginCollector.enabled_feature_groups({ProbeMultiA722B, ProbeMultiB722B})

    result = resolve_feature(MULTI_FEATURE, plugin_collector=collector)

    assert result.feature_group is None
    assert result.error is not None
    assert "Multiple FeatureGroups match feature name" in result.error
    assert "ProbeMultiA722B" in result.error
    assert "ProbeMultiB722B" in result.error
    # PINS CURRENT DIVERGENCE (#14): no module paths and no troubleshooting URL in the debug error.
    assert MODULE_PATH not in result.error
    assert TROUBLESHOOTING_URL not in result.error


# ---------------------------------------------------------------------------
# Divergence 15: not-found text
# ---------------------------------------------------------------------------


class ProbeNeedle722B(_ParityProbeBase722B):
    """Supports the needle name; feature_names_supported feeds the difflib suggestion."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {NEEDLE_FEATURE}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == NEEDLE_FEATURE


def test_engine_not_found_error_carries_suggestions_and_url() -> None:
    """The engine's not-found error suggests close names, points at resolve_feature, ends with the URL."""
    with pytest.raises(ValueError, match="No feature groups found") as exc_info:
        IdentifyFeatureGroupClass(
            feature=Feature(NEEDLE_TYPO),
            accessible_plugins={ProbeNeedle722B: {CfwParity722B}},
            links=None,
        )

    message = str(exc_info.value)
    # TARGET CONTRACT: suggestion, debug pointer, URL last.
    assert "Did you mean" in message
    assert NEEDLE_FEATURE in message
    assert "Use resolve_feature(" in message
    assert message.endswith(TROUBLESHOOTING_URL)


def test_resolve_feature_not_found_error_is_single_bare_line() -> None:
    """resolve_feature's not-found error is one bare line without suggestions or URL."""
    collector = PluginCollector.enabled_feature_groups({ProbeNeedle722B})

    result = resolve_feature(NEEDLE_TYPO, plugin_collector=collector)

    assert result.feature_group is None
    assert result.error is not None
    # PINS CURRENT DIVERGENCE (#15): the engine points users at a tool returning strictly less information.
    assert result.error == f"No FeatureGroup found for feature name: {NEEDLE_TYPO}"
    assert "Did you mean" not in result.error
    assert TROUBLESHOOTING_URL not in result.error
