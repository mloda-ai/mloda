import pytest
from typing import Set, Type

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.components.utils import get_all_subclasses
from mloda_core.abstract_plugins.components.validators.feature_validator import FeatureValidator


class TestValidateAndResolveComputeFramework:
    """Test the validate_and_resolve_compute_framework static method."""

    def test_valid_framework_name_returns_subclass(self) -> None:
        """Should return the matching ComputeFrameWork subclass when a valid framework name is provided."""
        available_frameworks = get_all_subclasses(ComputeFrameWork)

        # Get a valid framework name
        valid_framework = next(iter(available_frameworks))
        framework_name = valid_framework.get_class_name()

        # Act
        result = FeatureValidator.validate_and_resolve_compute_framework(
            framework_name=framework_name, available_frameworks=available_frameworks, source="parameter"
        )

        # Assert
        assert result == valid_framework
        assert issubclass(result, ComputeFrameWork)

    def test_invalid_framework_name_raises_value_error(self) -> None:
        """Should raise ValueError when framework name doesn't match any available framework."""
        available_frameworks = get_all_subclasses(ComputeFrameWork)
        invalid_framework_name = "NonExistentFramework"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            FeatureValidator.validate_and_resolve_compute_framework(
                framework_name=invalid_framework_name, available_frameworks=available_frameworks, source="parameter"
            )

        # Verify error message contains the framework name
        assert invalid_framework_name in str(exc_info.value)

    def test_empty_available_frameworks_raises_value_error(self) -> None:
        """Should raise ValueError when no frameworks are available."""
        empty_frameworks: Set[Type[ComputeFrameWork]] = set()

        # Act & Assert
        with pytest.raises(ValueError):
            FeatureValidator.validate_and_resolve_compute_framework(
                framework_name="AnyFramework", available_frameworks=empty_frameworks, source="parameter"
            )

    def test_error_message_includes_source_parameter(self) -> None:
        """Should include 'parameter' source in error message when framework not found."""
        available_frameworks = get_all_subclasses(ComputeFrameWork)
        invalid_framework_name = "InvalidFramework"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            FeatureValidator.validate_and_resolve_compute_framework(
                framework_name=invalid_framework_name, available_frameworks=available_frameworks, source="parameter"
            )

        # Verify error message includes the source
        assert "parameter" in str(exc_info.value)

    def test_error_message_includes_source_options(self) -> None:
        """Should include 'options' source in error message when framework not found."""
        available_frameworks = get_all_subclasses(ComputeFrameWork)
        invalid_framework_name = "InvalidFramework"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            FeatureValidator.validate_and_resolve_compute_framework(
                framework_name=invalid_framework_name, available_frameworks=available_frameworks, source="options"
            )

        # Verify error message includes the source
        assert "options" in str(exc_info.value)

    def test_case_sensitive_framework_matching(self) -> None:
        """Should match framework names case-sensitively."""
        available_frameworks = get_all_subclasses(ComputeFrameWork)

        if available_frameworks:
            valid_framework = next(iter(available_frameworks))
            framework_name = valid_framework.get_class_name()

            # Try with different case
            wrong_case_name = framework_name.lower() if framework_name[0].isupper() else framework_name.upper()

            # Should not match if case is different
            if wrong_case_name != framework_name:
                with pytest.raises(ValueError):
                    FeatureValidator.validate_and_resolve_compute_framework(
                        framework_name=wrong_case_name, available_frameworks=available_frameworks, source="parameter"
                    )


class TestValidateComputeFrameworksResolved:
    """Test the validate_compute_frameworks_resolved static method."""

    def test_none_raises_value_error(self) -> None:
        """Should raise ValueError when compute_frameworks is None."""
        # Act & Assert
        with pytest.raises(ValueError):
            FeatureValidator.validate_compute_frameworks_resolved(compute_frameworks=None, feature_name="TestFeature")

    def test_empty_set_does_not_raise(self) -> None:
        """Empty set is valid (different from None) and should not raise."""
        empty_frameworks: Set[Type[ComputeFrameWork]] = set()

        # Act & Assert - should not raise
        FeatureValidator.validate_compute_frameworks_resolved(
            compute_frameworks=empty_frameworks, feature_name="TestFeature"
        )

    def test_error_message_includes_feature_name(self) -> None:
        """Should include the feature name in error message."""
        feature_name = "MyTestFeature"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            FeatureValidator.validate_compute_frameworks_resolved(compute_frameworks=None, feature_name=feature_name)

        # Verify error message includes the feature name
        assert feature_name in str(exc_info.value)

    def test_valid_set_does_not_raise(self) -> None:
        """Should not raise when compute_frameworks is populated with valid frameworks."""
        available_frameworks = get_all_subclasses(ComputeFrameWork)

        # Act & Assert - should not raise
        FeatureValidator.validate_compute_frameworks_resolved(
            compute_frameworks=available_frameworks, feature_name="TestFeature"
        )

    def test_error_message_mentions_resolution(self) -> None:
        """Error message should indicate that frameworks need to be resolved."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            FeatureValidator.validate_compute_frameworks_resolved(compute_frameworks=None, feature_name="TestFeature")

        # Verify error message mentions resolution or similar concept
        error_msg = str(exc_info.value).lower()
        assert "resolved" in error_msg or "resolve" in error_msg or "framework" in error_msg
