"""check_subclass_scope: a well-scoped subclass passes, each broken pair fails only its own sub-check."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from mloda._testing.subclass_scope import assert_subclass_scope, check_subclass_scope
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.feature_group import FeatureGroup

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
    class KeyedParent(FeatureGroup):
        PROPERTY_MAPPING = {"variant": PropertySpec("Variant.", allowed_values=("x", "y"), strict_validation=True)}

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"kp_a"}

    class KeyedChild(KeyedParent):
        pass

    result = check_subclass_scope(KeyedChild, KeyedParent, SELECTING)
    assert not result.passed
    assert all(f.startswith("setup:") for f in result.failures), result.failures


def test_parent_without_names_needs_explicit_requests() -> None:
    parent, sub = make_nameless_pair()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert all(f.startswith("names:") for f in result.failures), result.failures


def test_explicit_requests_pass_for_nameless_parent() -> None:
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

    assert check_subclass_scope(RequestChild, RequestParent, SELECTING, requests=["rq_a"]).failures == ()


def test_assert_passes_silently_for_well_scoped_subclass() -> None:
    parent, sub = make_well_scoped()
    assert_subclass_scope(sub, parent, SELECTING)


def test_assert_raises_pytest_failure_for_ungated_subclass() -> None:
    parent, sub = make_ungated()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_subclass_scope(sub, parent, SELECTING)
    assert "steals:" in str(excinfo.value)
