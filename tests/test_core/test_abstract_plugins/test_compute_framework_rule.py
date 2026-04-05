"""Tests for compute_framework_rule() and compute_framework_definition() contract.

Validates that the API contract is clear and consistent:
- compute_framework_rule() returns None (all frameworks) or a set of specific frameworks.
- compute_framework_definition() resolves the rule into a concrete set.
- Invalid return types raise ValueError with a helpful message.
"""

from unittest.mock import patch

import pytest

from mloda.provider import ComputeFramework, FeatureGroup
from mloda.core.abstract_plugins.components.utils import get_all_subclasses


class StubComputeFramework(ComputeFramework):
    """Minimal compute framework for testing."""

    @classmethod
    def expected_data_framework(cls) -> type:
        return dict


class AnotherStubComputeFramework(ComputeFramework):
    """Second minimal compute framework for testing."""

    @classmethod
    def expected_data_framework(cls) -> type:
        return list


class _ValidTestGroup(FeatureGroup):
    """A valid FeatureGroup subclass used only for testing invalid rule returns via mocking."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {StubComputeFramework}


class TestComputeFrameworkRuleBaseClassDefault:
    """The base class default should return None, meaning 'all frameworks'."""

    def test_base_class_returns_none(self) -> None:
        assert FeatureGroup.compute_framework_rule() is None

    def test_none_means_all_frameworks(self) -> None:
        """compute_framework_definition() should resolve None to all known subclasses."""
        result = FeatureGroup.compute_framework_definition()
        all_frameworks = get_all_subclasses(ComputeFramework)
        assert result == all_frameworks
        assert len(result) > 0


class TestComputeFrameworkDefinitionWithExplicitSet:
    """When a plugin returns a specific set, it should pass through unchanged."""

    def test_single_framework_passthrough(self) -> None:
        assert _ValidTestGroup.compute_framework_definition() == {StubComputeFramework}

    def test_multiple_frameworks_passthrough(self) -> None:
        with patch.object(
            _ValidTestGroup,
            "compute_framework_rule",
            return_value={StubComputeFramework, AnotherStubComputeFramework},
        ):
            result = _ValidTestGroup.compute_framework_definition()
            assert result == {StubComputeFramework, AnotherStubComputeFramework}


class TestComputeFrameworkDefinitionRejectsInvalidTypes:
    """Invalid return types should raise ValueError with a helpful message.

    Uses mock.patch to temporarily override compute_framework_rule on a valid subclass,
    avoiding creation of broken FeatureGroup subclasses that pollute the subclass registry.
    """

    def test_bool_true_raises_valueerror(self) -> None:
        with patch.object(_ValidTestGroup, "compute_framework_rule", return_value=True):
            with pytest.raises(ValueError, match="must return None or a set"):
                _ValidTestGroup.compute_framework_definition()

    def test_bool_false_raises_valueerror(self) -> None:
        with patch.object(_ValidTestGroup, "compute_framework_rule", return_value=False):
            with pytest.raises(ValueError, match="must return None or a set"):
                _ValidTestGroup.compute_framework_definition()

    def test_string_raises_valueerror(self) -> None:
        with patch.object(_ValidTestGroup, "compute_framework_rule", return_value="PandasDataFrame"):
            with pytest.raises(ValueError, match="must return None or a set"):
                _ValidTestGroup.compute_framework_definition()

    def test_list_raises_valueerror(self) -> None:
        with patch.object(_ValidTestGroup, "compute_framework_rule", return_value=[StubComputeFramework]):
            with pytest.raises(ValueError, match="must return None or a set"):
                _ValidTestGroup.compute_framework_definition()

    def test_error_message_includes_class_name(self) -> None:
        with patch.object(_ValidTestGroup, "compute_framework_rule", return_value=42):
            with pytest.raises(ValueError, match="_ValidTestGroup"):
                _ValidTestGroup.compute_framework_definition()

    def test_error_message_includes_actual_type(self) -> None:
        with patch.object(_ValidTestGroup, "compute_framework_rule", return_value=42):
            with pytest.raises(ValueError, match="int"):
                _ValidTestGroup.compute_framework_definition()


class TestComputeFrameworkRuleReturnTypeConsistency:
    """Override return types should be a strict subset of the base class contract."""

    def test_override_returning_set_is_subset_of_all(self) -> None:
        """A plugin restricting to a specific set should be a subset of all frameworks."""
        restricted = _ValidTestGroup.compute_framework_definition()
        all_frameworks = FeatureGroup.compute_framework_definition()
        assert restricted.issubset(all_frameworks)

    def test_override_returning_none_equals_all(self) -> None:
        """A plugin returning None should resolve to all frameworks, same as the base class."""
        with patch.object(_ValidTestGroup, "compute_framework_rule", return_value=None):
            result = _ValidTestGroup.compute_framework_definition()
            base_result = FeatureGroup.compute_framework_definition()
            assert result == base_result
