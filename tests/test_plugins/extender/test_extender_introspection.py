"""Tests for introspection helpers on the Extender base class (issue #572).

These helpers let custom extenders (tracing/otel/lineage) resolve the feature
group name, FeatureSet, feature name, and options from the hook invocation
`extender(feature_group.calculate_feature, data, features)` without every
plugin reimplementing bound-method introspection.
"""

import functools
from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.function_extender import (
    Extender,
    ExtenderHook,
    _CompositeExtender,
)


class SampleFeatureGroup:
    """FeatureGroup-like class exposing calculate_feature as a classmethod, as core invokes it."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    def instance_method(self, data: Any, features: FeatureSet) -> Any:
        return data

    @staticmethod
    def static_method(data: Any, features: FeatureSet) -> Any:
        return data


def top_level_function(data: Any, features: FeatureSet) -> Any:
    return data


class RecordingExtender(Extender):
    """Extender that records the resolved feature group name on every call."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.seen_names: list[str] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.seen_names.append(self.feature_group_name(func))
        return func(*args, **kwargs)


def build_feature_set() -> FeatureSet:
    return FeatureSet([Feature("some_feature", {"key": "value"})])


def make_opaque_wrapper(inner: Any) -> Any:
    """Wrap without functools.wraps: only __wrapped__ links back to the inner callable."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return inner(*args, **kwargs)

    wrapper.__wrapped__ = inner  # type: ignore[attr-defined]
    return wrapper


class TestFeatureGroupName:
    """Extender.feature_group_name resolves the owning class name of the hooked callable."""

    def test_bound_classmethod_returns_class_name(self) -> None:
        assert Extender.feature_group_name(SampleFeatureGroup.calculate_feature) == "SampleFeatureGroup"

    def test_bound_instance_method_returns_instance_class_name(self) -> None:
        instance = SampleFeatureGroup()
        assert Extender.feature_group_name(instance.instance_method) == "SampleFeatureGroup"

    def test_unbound_function_with_dotted_qualname_returns_class_name(self) -> None:
        unbound = SampleFeatureGroup.__dict__["instance_method"]
        assert Extender.feature_group_name(unbound) == "SampleFeatureGroup"

    def test_staticmethod_returns_class_name_from_qualname(self) -> None:
        assert Extender.feature_group_name(SampleFeatureGroup.static_method) == "SampleFeatureGroup"

    def test_top_level_function_returns_unknown(self) -> None:
        assert Extender.feature_group_name(top_level_function) == "unknown"

    def test_object_without_self_and_qualname_returns_unknown(self) -> None:
        partial_func = functools.partial(top_level_function)
        assert not hasattr(partial_func, "__qualname__")
        assert not hasattr(partial_func, "__self__")
        assert Extender.feature_group_name(partial_func) == "unknown"

    def test_functools_wraps_wrapper_is_unwrapped_to_bound_classmethod(self) -> None:
        @functools.wraps(SampleFeatureGroup.calculate_feature)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return SampleFeatureGroup.calculate_feature(*args, **kwargs)

        assert Extender.feature_group_name(wrapper) == "SampleFeatureGroup"

    def test_opaque_wrapped_chain_is_unwrapped_before_inspection(self) -> None:
        # No functools.wraps: the outer closures have useless qualnames, only
        # the __wrapped__ chain leads back to the bound classmethod.
        once = make_opaque_wrapper(SampleFeatureGroup.calculate_feature)
        twice = make_opaque_wrapper(once)
        assert Extender.feature_group_name(twice) == "SampleFeatureGroup"


class TestFeatureSetHelper:
    """Extender.feature_set finds the FeatureSet in hook args."""

    def test_returns_feature_set_from_args(self) -> None:
        feature_set = build_feature_set()
        assert Extender.feature_set(({"col": [1]}, feature_set)) is feature_set

    def test_returns_first_feature_set_when_multiple(self) -> None:
        first = build_feature_set()
        second = build_feature_set()
        assert Extender.feature_set((first, second)) is first

    def test_returns_none_when_no_feature_set_in_args(self) -> None:
        assert Extender.feature_set(({"col": [1]}, "not_a_feature_set")) is None

    def test_returns_none_for_empty_args(self) -> None:
        assert Extender.feature_set(()) is None


class TestFeatureNameHelper:
    """Extender.feature_name resolves one feature name from the FeatureSet in hook args."""

    def test_returns_feature_name_string(self) -> None:
        feature_set = build_feature_set()
        result = Extender.feature_name(({"col": [1]}, feature_set))
        assert result == "some_feature"
        assert isinstance(result, str)

    def test_returns_none_when_no_feature_set_in_args(self) -> None:
        assert Extender.feature_name(({"col": [1]},)) is None

    def test_returns_none_for_empty_feature_set_without_raising(self) -> None:
        empty_feature_set = FeatureSet()
        assert Extender.feature_name(({"col": [1]}, empty_feature_set)) is None


class TestFeatureOptionsHelper:
    """Extender.feature_options resolves the Options of the FeatureSet in hook args."""

    def test_returns_options_of_feature_set(self) -> None:
        feature_set = build_feature_set()
        result = Extender.feature_options(({"col": [1]}, feature_set))
        assert isinstance(result, Options)
        assert result.get("key") == "value"

    def test_returns_none_when_no_feature_set_in_args(self) -> None:
        assert Extender.feature_options(({"col": [1]},)) is None

    def test_returns_none_when_options_not_set(self) -> None:
        empty_feature_set = FeatureSet()
        assert empty_feature_set.options is None
        assert Extender.feature_options(({"col": [1]}, empty_feature_set)) is None


class TestCompositeExtenderNamePropagation:
    """Regression: every extender in a composite chain must see the real feature group name.

    Today _CompositeExtender.make_wrapper hands all but the innermost extender an
    anonymous `wrapper` closure, so name introspection breaks for the outer ones.
    """

    def test_all_extenders_in_chain_resolve_real_class_name(self) -> None:
        ext_a = RecordingExtender(priority=10)
        ext_b = RecordingExtender(priority=20)
        composite = _CompositeExtender([ext_a, ext_b])

        feature_set = build_feature_set()
        result = composite(SampleFeatureGroup.calculate_feature, {"x": [1]}, feature_set)

        assert result == {"x": [1]}
        assert ext_a.seen_names == ["SampleFeatureGroup"], (
            "First extender in the chain must resolve the real feature group class name"
        )
        assert ext_b.seen_names == ["SampleFeatureGroup"], (
            "Second extender in the chain must resolve the real feature group class name"
        )
