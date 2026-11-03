"""check_subclass_scope: a well-scoped subclass passes, each broken pair fails only its own sub-check."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from mloda._testing.subclass_scope import assert_subclass_scope, check_subclass_scope
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import PreFilterPlugins

Pair = tuple[type[FeatureGroup], type[FeatureGroup]]
SELECTING = Options(group={"variant": "x"})


def make_parent() -> type[FeatureGroup]:
    class ScopeParent(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"sc_a"}

    return ScopeParent


def _gated_subclass(parent: type[FeatureGroup], required_key: str) -> type[FeatureGroup]:
    class GatedChild(parent):  # type: ignore[valid-type, misc]
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            if required_key not in options.group and required_key not in options.context:
                return False
            return super().match_feature_group_criteria(feature_name, options, data_access_collection)  # type: ignore[no-any-return]

    return GatedChild


def make_well_scoped() -> Pair:
    parent = make_parent()
    return parent, _gated_subclass(parent, "variant")


def make_ungated() -> Pair:
    parent = make_parent()

    class UngatedChild(parent):  # type: ignore[valid-type, misc]
        pass

    return parent, UngatedChild


def make_gate_rejects_own_option() -> Pair:
    parent = make_parent()
    return parent, _gated_subclass(parent, "other")


def make_unrelated() -> Pair:
    return make_parent(), make_well_scoped()[1]


def make_nameless_pair() -> Pair:
    class NamelessParent(FeatureGroup):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "nl_a"

    class NamelessChild(NamelessParent):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "nl_a" and "variant" in options.group

    return NamelessParent, NamelessChild


def make_keyed_pair() -> Pair:
    class KeyedParent(FeatureGroup):
        PROPERTY_MAPPING = {"variant": PropertySpec("Variant.", allowed_values=("x", "y"), strict_validation=True)}

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"kp_a"}

    class KeyedChild(KeyedParent):
        pass

    return KeyedParent, KeyedChild


def make_request_pair() -> Pair:
    class RequestParent(FeatureGroup):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "rq_a"

    class RequestChild(RequestParent):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "rq_a" and "variant" in options.group

    return RequestParent, RequestChild


def make_optional_option_stealer() -> Pair:
    class OptParent(FeatureGroup):
        PROPERTY_MAPPING = {"opt": PropertySpec("Optional.", allowed_values=("a", "b"), default=None)}

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"op_a"}

    class OptStealer(OptParent):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            if not {"opt", "variant"} & {*options.group, *options.context}:
                return False
            return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    return OptParent, OptStealer


def make_parent_rejecting_selecting_option() -> Pair:
    class StrictParent(FeatureGroup):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "sp_a" and "variant" not in options.group

    class StrictChild(StrictParent):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "sp_a" and "variant" in options.group

    return StrictParent, StrictChild


def make_unloaded_framework_pair() -> tuple[type[FeatureGroup], type[FeatureGroup], type[ComputeFramework]]:
    class UnloadedFramework(ComputeFramework):
        pass

    parent = make_parent()

    class UnloadedChild(parent):  # type: ignore[valid-type, misc]
        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {UnloadedFramework}

    return parent, UnloadedChild, UnloadedFramework


def test_well_scoped_subclass_passes() -> None:
    parent, sub = make_well_scoped()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert result.check == "subclass_scope"
    assert result.subject == sub.__name__
    assert result.failures == ()


@pytest.mark.parametrize(
    ("make", "prefix"),
    [
        (make_ungated, "steals:"),
        (make_gate_rejects_own_option, "misses:"),
    ],
)
def test_broken_pair_fails_only_its_sub_check(make: Callable[[], Pair], prefix: str) -> None:
    parent, sub = make()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert all(f.startswith(prefix) for f in result.failures), result.failures


def test_not_a_subclass_is_a_setup_failure() -> None:
    parent, sub = make_unrelated()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert all(f.startswith("setup:") for f in result.failures), result.failures


def test_selecting_key_already_in_base_options_is_a_setup_failure() -> None:
    parent, sub = make_keyed_pair()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert all(f.startswith("setup:") for f in result.failures), result.failures


def test_parent_without_names_needs_explicit_requests() -> None:
    parent, sub = make_nameless_pair()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert all(f.startswith("names:") for f in result.failures), result.failures


def test_explicit_requests_pass_for_nameless_parent() -> None:
    parent, sub = make_request_pair()
    assert check_subclass_scope(sub, parent, SELECTING, requests=["rq_a"]).failures == ()


def test_partial_steal_by_optional_parent_option_fails_only_steals() -> None:
    parent, sub = make_optional_option_stealer()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert all(f.startswith("steals:") for f in result.failures), result.failures


def test_subclass_serving_request_alone_when_parent_rejects_selecting_option_passes() -> None:
    parent, sub = make_parent_rejecting_selecting_option()
    assert check_subclass_scope(sub, parent, SELECTING, requests=["sp_a"]).failures == ()


def test_unloaded_compute_framework_is_a_setup_failure_naming_the_class(monkeypatch: pytest.MonkeyPatch) -> None:
    parent, sub, dummy = make_unloaded_framework_pair()
    loaded = PreFilterPlugins.get_cfw_subclasses()
    monkeypatch.setattr(
        PreFilterPlugins, "get_cfw_subclasses", staticmethod(lambda: {c for c in loaded if c is not dummy})
    )
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert any(f.startswith("setup:") and sub.__name__ in f for f in result.failures), result.failures


def test_parent_rejecting_its_own_request_is_a_setup_failure() -> None:
    parent, sub = make_request_pair()
    result = check_subclass_scope(sub, parent, SELECTING, requests=["not_rq_a"])
    assert not result.passed
    assert any(f.startswith("setup:") for f in result.failures), result.failures


def test_selecting_option_in_context_passes_for_gated_subclass() -> None:
    parent, sub = make_well_scoped()
    assert check_subclass_scope(sub, parent, Options(context={"variant": "x"})).failures == ()


def test_explicitly_empty_requests_is_a_names_failure_without_pass_requests_hint() -> None:
    parent, sub = make_well_scoped()
    result = check_subclass_scope(sub, parent, SELECTING, requests=[])
    assert not result.passed
    assert all(f.startswith("names:") for f in result.failures), result.failures
    assert all("pass requests=" not in f for f in result.failures), result.failures


def test_assert_passes_silently_for_well_scoped_subclass() -> None:
    parent, sub = make_well_scoped()
    assert_subclass_scope(sub, parent, SELECTING)


def test_assert_raises_pytest_failure_for_ungated_subclass() -> None:
    parent, sub = make_ungated()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_subclass_scope(sub, parent, SELECTING)
    assert "steals:" in str(excinfo.value)
