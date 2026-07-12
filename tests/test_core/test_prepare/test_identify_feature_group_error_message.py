"""Tests for error message formatting in identify_feature_group.py.

This test verifies that error messages use formatted output from
format_feature_group_classes instead of raw dict/class representation.
"""

from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


class MockComputeFramework(ComputeFramework):
    """Mock compute framework for testing."""

    pass


class ConflictingFeatureGroupA(FeatureGroup):
    """First conflicting feature group with custom domain 'domain_a'.

    This feature group matches any feature named 'conflicting_test_feature'.
    """

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("domain_a")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == "conflicting_test_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ConflictingFeatureGroupB(FeatureGroup):
    """Second conflicting feature group with custom domain 'domain_b'.

    This feature group also matches any feature named 'conflicting_test_feature',
    creating a conflict scenario.
    """

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("domain_b")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == "conflicting_test_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestIdentifyFeatureGroupErrorMessageFormat:
    """Tests for the error message format when multiple feature groups are found."""

    def test_error_message_contains_formatted_class_names(self) -> None:
        """Test that the ValueError message contains formatted class names.

        When multiple feature groups match the same feature, the error message
        should contain formatted class names like:
          - ConflictingFeatureGroupA (module.path) [domain: domain_a]
          - ConflictingFeatureGroupB (module.path) [domain: domain_b]
        """
        feature = Feature("conflicting_test_feature")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            ConflictingFeatureGroupA: {MockComputeFramework},
            ConflictingFeatureGroupB: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "  - ConflictingFeatureGroupA" in error_message, (
            f"Error message should contain formatted class name '  - ConflictingFeatureGroupA', "
            f"but got: {error_message}"
        )

    def test_error_message_contains_module_path_in_parentheses(self) -> None:
        """Test that the error message contains module path in parentheses.

        Expected format includes module path like:
          - ClassName (tests.test_core.test_prepare.test_identify_feature_group_error_message)
        """
        feature = Feature("conflicting_test_feature")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            ConflictingFeatureGroupA: {MockComputeFramework},
            ConflictingFeatureGroupB: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "(tests.test_core.test_prepare.test_identify_feature_group_error_message)" in error_message, (
            f"Error message should contain module path in parentheses, but got: {error_message}"
        )

    def test_error_message_contains_domain_info(self) -> None:
        """Test that the error message contains domain information.

        Expected format includes domain like:
          - ClassName (module.path) [domain: domain_a]
        """
        feature = Feature("conflicting_test_feature")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            ConflictingFeatureGroupA: {MockComputeFramework},
            ConflictingFeatureGroupB: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "[domain: domain_a]" in error_message, (
            f"Error message should contain '[domain: domain_a]', but got: {error_message}"
        )
        assert "[domain: domain_b]" in error_message, (
            f"Error message should contain '[domain: domain_b]', but got: {error_message}"
        )

    def test_error_message_does_not_contain_raw_dict_representation(self) -> None:
        """Test that the error message does NOT contain raw dict/class representation.

        The old format looks like:
            Multiple feature groups {<class '...'>: {<class '...'>}} found for feature name: ...

        This raw representation should NOT appear in the new format.
        """
        feature = Feature("conflicting_test_feature")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            ConflictingFeatureGroupA: {MockComputeFramework},
            ConflictingFeatureGroupB: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "{<class" not in error_message, (
            f"Error message should NOT contain raw class representation '{{<class', but got: {error_message}"
        )

    def test_error_message_contains_feature_name(self) -> None:
        """Test that the error message mentions the feature name."""
        feature = Feature("conflicting_test_feature")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            ConflictingFeatureGroupA: {MockComputeFramework},
            ConflictingFeatureGroupB: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "'conflicting_test_feature'" in error_message, (
            f"Error message should contain feature name in quotes, but got: {error_message}"
        )


class KnownFeatureGroup(FeatureGroup):
    """Feature group that matches 'known_feature' for testing fuzzy match suggestions."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"known_feature", "another_feature"}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == "known_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestNoFeatureGroupFoundErrorMessage:
    """Tests for the improved 'no feature groups found' error message."""

    def test_no_plugins_loaded_suggests_plugin_loader(self) -> None:
        feature = Feature("some_feature")
        accessible_plugins: FeatureGroupEnvironmentMapping = {}

        with pytest.raises(ValueError, match="PluginLoader.all"):
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
            )

    def test_suggests_similar_feature_names(self) -> None:
        feature = Feature("knwon_feature")  # typo of "known_feature"
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            KnownFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError, match="Did you mean") as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
            )

        error_message = str(exc_info.value)
        assert "known_feature" in error_message

    def test_includes_troubleshooting_link(self) -> None:
        feature = Feature("nonexistent_xyz")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            KnownFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError, match="troubleshooting"):
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
            )

    def test_suggests_resolve_feature(self) -> None:
        feature = Feature("nonexistent_xyz")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            KnownFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError, match="resolve_feature"):
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
            )

    def test_no_similar_names_still_shows_help(self) -> None:
        feature = Feature("zzz_completely_different")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            KnownFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
            )

        error_message = str(exc_info.value)
        assert "resolve_feature" in error_message
        assert "Did you mean" not in error_message


class NoComputeFrameworkFeatureGroup(FeatureGroup):
    """Feature group for testing 'no compute framework' error message."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == "no_compute_framework_test_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestNoComputeFrameworkErrorMessageFormat:
    """Tests for the error message format when a feature group has no compute framework.

    The 'no compute framework' error is raised in the validate() method when a
    single feature group is found but its compute_frameworks set is empty.

    We use unittest.mock to patch the _filter_loop method to return a feature
    group with empty compute_frameworks, since the normal filtering logic
    excludes feature groups with no frameworks.
    """

    def test_no_compute_framework_error_contains_module_path(self) -> None:
        """Test that the 'no compute framework' error includes module path.

        The error message should contain the module path in parentheses like:
          NoComputeFrameworkFeatureGroup (tests.test_core.test_prepare.test_identify_feature_group_error_message)
        """
        from unittest.mock import patch

        feature = Feature("no_compute_framework_test_feature")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            NoComputeFrameworkFeatureGroup: {MockComputeFramework},
        }

        def mock_filter_loop(
            self: IdentifyFeatureGroupClass, *args: object, **kwargs: object
        ) -> FeatureGroupEnvironmentMapping:
            return {NoComputeFrameworkFeatureGroup: set()}

        with patch.object(IdentifyFeatureGroupClass, "_filter_loop", mock_filter_loop):
            with pytest.raises(ValueError) as exc_info:
                IdentifyFeatureGroupClass(
                    feature=feature,
                    accessible_plugins=accessible_plugins,
                    links=None,
                    data_access_collection=None,
                )

        error_message = str(exc_info.value)

        assert "no compute framework" in error_message.lower(), (
            f"Error should be about 'no compute framework', but got: {error_message}"
        )
        assert "(tests.test_core.test_prepare.test_identify_feature_group_error_message)" in error_message, (
            f"Error message should contain module path in parentheses, but got: {error_message}"
        )

    def test_no_compute_framework_error_contains_class_name(self) -> None:
        """Test that the 'no compute framework' error includes the class name.

        The error message should contain the class name.
        """
        from unittest.mock import patch

        feature = Feature("no_compute_framework_test_feature")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            NoComputeFrameworkFeatureGroup: {MockComputeFramework},
        }

        def mock_filter_loop(
            self: IdentifyFeatureGroupClass, *args: object, **kwargs: object
        ) -> FeatureGroupEnvironmentMapping:
            return {NoComputeFrameworkFeatureGroup: set()}

        with patch.object(IdentifyFeatureGroupClass, "_filter_loop", mock_filter_loop):
            with pytest.raises(ValueError) as exc_info:
                IdentifyFeatureGroupClass(
                    feature=feature,
                    accessible_plugins=accessible_plugins,
                    links=None,
                    data_access_collection=None,
                )

        error_message = str(exc_info.value)

        assert "no compute framework" in error_message.lower(), (
            f"Error should be about 'no compute framework', but got: {error_message}"
        )
        assert "NoComputeFrameworkFeatureGroup" in error_message, (
            f"Error message should contain class name 'NoComputeFrameworkFeatureGroup', but got: {error_message}"
        )

    def test_no_compute_framework_error_uses_formatted_pattern(self) -> None:
        """Test that the error uses the 'ClassName (module.path)' format.

        The error should use format_feature_group_class() which returns:
          ClassName (module.path)
        """
        from unittest.mock import patch

        feature = Feature("no_compute_framework_test_feature")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            NoComputeFrameworkFeatureGroup: {MockComputeFramework},
        }

        def mock_filter_loop(
            self: IdentifyFeatureGroupClass, *args: object, **kwargs: object
        ) -> FeatureGroupEnvironmentMapping:
            return {NoComputeFrameworkFeatureGroup: set()}

        with patch.object(IdentifyFeatureGroupClass, "_filter_loop", mock_filter_loop):
            with pytest.raises(ValueError) as exc_info:
                IdentifyFeatureGroupClass(
                    feature=feature,
                    accessible_plugins=accessible_plugins,
                    links=None,
                    data_access_collection=None,
                )

        error_message = str(exc_info.value)

        assert "no compute framework" in error_message.lower(), (
            f"Error should be about 'no compute framework', but got: {error_message}"
        )

        expected_pattern = (
            "NoComputeFrameworkFeatureGroup (tests.test_core.test_prepare.test_identify_feature_group_error_message)"
        )
        assert expected_pattern in error_message, (
            f"Error message should contain formatted class '{expected_pattern}', but got: {error_message}"
        )


class KnowledgeGraphLikeFeatureGroup(FeatureGroup):
    """Strict input-feature matcher: accepts the bare name 'knowledge_graph' but
    rejects any feature that carries extra group options.

    This mimics a knowledge-graph style feature group that is meant to be an
    input feature. When a parent feature forwards its query-specific group
    options onto this declared input feature, the extra group keys cause the
    matcher to reject the feature entirely.
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        name = str(feature_name) if isinstance(feature_name, FeatureName) else feature_name
        return name == "knowledge_graph" and not options.group

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class OtherNameFeatureGroup(FeatureGroup):
    """Feature group that only matches a different name, used to keep some
    plugins accessible while never matching the feature under test."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        name = str(feature_name) if isinstance(feature_name, FeatureName) else feature_name
        return name == "some_other_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestInputFeatureForwardingHint:
    """Tests for the forwarding hint in the no-feature-group error.

    The hint should fire only when a feature fails to resolve BECAUSE its group
    options carry keys the matcher rejects, and some accessible feature group
    WOULD match the same bare feature name if the group options were empty.
    Under the forward-by-default contract, group options flow onto input
    features from the consumer BY DEFAULT, so the hint must say so and advise
    the remedies: forward_group_exclude, an allowlist, or forward_group=False
    on the child in the consumer's input_features.
    """

    def test_hint_names_offending_keys_and_helper(self) -> None:
        feature = Feature("knowledge_graph", Options(group={"query_text": "hi", "top_k": 5}))

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            KnowledgeGraphLikeFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "forward_group" in error_message, (
            f"Error message should point to forward_group, but got: {error_message}"
        )
        assert "query_text" in error_message, (
            f"Error message should name offending key 'query_text', but got: {error_message}"
        )
        assert "top_k" in error_message, f"Error message should name offending key 'top_k', but got: {error_message}"
        assert "KnowledgeGraphLikeFeatureGroup" in error_message, (
            f"Error message should name culprit feature group, but got: {error_message}"
        )
        # Forward-by-default contract: the hint must state that group options flow
        # onto input features from the consumer by default, and advise the new
        # remedies instead of the retired allowlist-first phrasing.
        assert "by default" in error_message.lower(), (
            f"Error message should state that group options flow onto input features by default, "
            f"but got: {error_message}"
        )
        assert "forward_group_exclude" in error_message, (
            f"Error message should advise the forward_group_exclude remedy, but got: {error_message}"
        )
        assert "forward_group=False" in error_message, (
            f"Error message should advise the forward_group=False remedy, but got: {error_message}"
        )
        assert "stop using forward_group=True" not in error_message, (
            f"Error message must not contain the retired phrasing 'stop using forward_group=True', "
            f"but got: {error_message}"
        )

    def test_no_hint_when_no_group_options(self) -> None:
        feature = Feature("knowledge_graph")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            OtherNameFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "forward_group" not in error_message, (
            f"Error message should NOT emit forwarding hint when no group options, but got: {error_message}"
        )

    def test_no_hint_when_no_feature_group_matches_bare_name(self) -> None:
        feature = Feature("totally_unknown", Options(group={"query_text": "hi"}))

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            KnowledgeGraphLikeFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "forward_group" not in error_message, (
            f"Error message should NOT emit forwarding hint when no FG matches the bare name, but got: {error_message}"
        )

    def test_reserved_keys_not_listed_as_offending(self) -> None:
        feature = Feature(
            "knowledge_graph",
            Options(
                group={
                    "query_text": "hi",
                    DefaultOptionKeys.in_features: "some_source",
                }
            ),
        )

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            KnowledgeGraphLikeFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "forward_group" in error_message, (
            f"Error message should point to forward_group, but got: {error_message}"
        )
        assert "query_text" in error_message, (
            f"Error message should name offending key 'query_text', but got: {error_message}"
        )
        assert "in_features" not in error_message, (
            f"Reserved key 'in_features' must NOT be listed as offending, but got: {error_message}"
        )


class StrictWindowFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Config-based feature group with a strict, validated 'window_size' property.

    Deliberately does NOT override match_feature_group_criteria, so it exercises the
    mixin's default swallow-to-False path: an element_validator rejection inside
    FeatureChainParser._validate_property_value raises ValueError, which
    match_feature_group_criteria catches and turns into a plain False, discarding the
    actionable message. _strict_validation_rejection_reason recovers that message.
    """

    PROPERTY_MAPPING = {
        "window_size": {
            "explanation": "Size of window",
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.element_validator: lambda v: isinstance(v, int) and 0 < v <= 13,
        },
        DefaultOptionKeys.in_features: {"explanation": "source", DefaultOptionKeys.context: True},
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class StrictMaxWindowFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Second strict-rejecting config-based feature group, mapping a different key
    ('max_window') than StrictWindowFeatureGroup, used to verify the hint names
    every rejecting candidate rather than only the first.
    """

    PROPERTY_MAPPING = {
        "max_window": {
            "explanation": "Maximum window size",
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.element_validator: lambda v: isinstance(v, int) and 0 < v <= 13,
        },
        DefaultOptionKeys.in_features: {"explanation": "source", DefaultOptionKeys.context: True},
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestStrictValidationRejectionHint:
    """Tests for the strict-validation rejection hint in the no-feature-group error.

    The no-feature-group-found error consults
    FeatureChainParserMixin._strict_validation_rejection_reason for each accessible
    candidate and surfaces the discarded ValueError (e.g. "Property value '14' failed
    validation for 'window_size'") together with the culprit class name(s).
    """

    def test_hint_names_rejected_option_value_and_class(self) -> None:
        feature = Feature(
            "strict_window_test_feature",
            Options(context={DefaultOptionKeys.in_features: "src", "window_size": 14}),
        )

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            StrictWindowFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "window_size" in error_message, (
            f"Error message should name the rejected option 'window_size', but got: {error_message}"
        )
        assert "14" in error_message, f"Error message should include the rejected value '14', but got: {error_message}"
        assert "StrictWindowFeatureGroup" in error_message, (
            f"Error message should name the culprit feature group, but got: {error_message}"
        )

    def test_no_hint_when_feature_is_unrelated(self) -> None:
        """A feature with none of the mapped keys present is a non-match, not a
        rejection: match_feature_group_criteria returns False without raising, so
        _strict_validation_rejection_reason must find nothing to surface."""
        feature = Feature("totally_unrelated_feature")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            StrictWindowFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "StrictWindowFeatureGroup" not in error_message, (
            f"Error message should NOT name a feature group for an unrelated feature, but got: {error_message}"
        )
        assert "window_size" not in error_message, (
            f"Error message should NOT mention 'window_size' for an unrelated feature, but got: {error_message}"
        )

    def test_hint_names_all_rejecting_classes(self) -> None:
        """When multiple accessible candidates each reject the same feature via strict
        validation, the hint must name every one of them, not just the first."""
        feature = Feature(
            "strict_multi_test_feature",
            Options(
                context={
                    DefaultOptionKeys.in_features: "src",
                    "window_size": 14,
                    "max_window": 20,
                }
            ),
        )

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            StrictWindowFeatureGroup: {MockComputeFramework},
            StrictMaxWindowFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "StrictWindowFeatureGroup" in error_message, (
            f"Error message should name StrictWindowFeatureGroup, but got: {error_message}"
        )
        assert "StrictMaxWindowFeatureGroup" in error_message, (
            f"Error message should name StrictMaxWindowFeatureGroup, but got: {error_message}"
        )
