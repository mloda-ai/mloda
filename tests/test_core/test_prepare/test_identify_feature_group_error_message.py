"""Tests for error message formatting in identify_feature_group.py.

This test verifies that error messages use formatted output from
format_feature_group_classes instead of raw dict/class representation.
"""

from typing import Optional, Set, Type, Union

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
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
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "conflicting_test_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
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
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "conflicting_test_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
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


class NoComputeFrameworkFeatureGroup(FeatureGroup):
    """Feature group for testing 'no compute framework' error message."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "no_compute_framework_test_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
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
