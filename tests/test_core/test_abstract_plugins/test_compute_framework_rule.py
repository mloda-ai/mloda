"""Tests for compute_framework_rule() and compute_framework_definition() contract.

Validates that the API contract is clear and consistent:
- compute_framework_rule() returns None (all frameworks) or a set of specific frameworks.
- compute_framework_definition() resolves the rule into a concrete set.
- Invalid return types raise ValueError with a helpful message.
"""

import gc
import weakref
from typing import Any
from unittest.mock import patch

import pytest

from mloda.provider import ComputeFramework, FeatureGroup
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup
from mloda.core.abstract_plugins.components.utils import get_all_subclasses

from tests.helpers.subclass_walks import count_root_walks


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

    def test_none_means_all_frameworks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """compute_framework_definition() resolves None to all known subclasses without walking the tree."""
        expected = get_all_subclasses(ComputeFramework)
        root_walks = count_root_walks(monkeypatch, ComputeFramework)
        results = [FeatureGroup.compute_framework_definition() for _ in range(3)]
        monkeypatch.undo()

        assert all(result == expected for result in results)
        assert len(expected) > 0
        assert root_walks == []

    def test_operation_group_definition_never_walks_the_compute_framework_tree(
        self, monkeypatch: pytest.MonkeyPatch, make_implementation: Any
    ) -> None:
        class _ServedGroup(OperationFeatureGroup):
            def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
                return None

        make_implementation(_ServedGroup, StubComputeFramework)
        root_walks = count_root_walks(monkeypatch, ComputeFramework)
        results = [_ServedGroup.compute_framework_definition() for _ in range(3)]
        monkeypatch.undo()

        assert all(result == {StubComputeFramework} for result in results)
        assert root_walks == []

    def test_definition_tracks_a_transient_subclass(self) -> None:
        """A new subclass shows up at once and a collected one disappears."""

        class _CooperativeMixin:
            def __init_subclass__(cls, **kwargs: Any) -> None:
                super().__init_subclass__(**kwargs)

        class TransientFramework(_CooperativeMixin, StubComputeFramework):
            @classmethod
            def expected_data_framework(cls) -> type:
                return set

        present = TransientFramework in FeatureGroup.compute_framework_definition()
        name = TransientFramework.__name__
        ref = weakref.ref(TransientFramework)
        del TransientFramework
        for generation in (0, 1, 2):
            if ref() is None:
                break
            gc.collect(generation)
        collected = ref() is None
        names = {c.__name__ for c in FeatureGroup.compute_framework_definition()}

        assert present
        assert collected
        assert name not in names


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
