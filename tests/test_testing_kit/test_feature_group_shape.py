"""check_feature_group_shape: a conforming plain FeatureGroup passes, each broken one fails its own sub-check."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from mloda._testing.feature_group_shape import assert_feature_group_shape, check_feature_group_shape
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda_plugins.feature_group.experimental.environment.installed_packages_feature_group import (
    InstalledPackagesFeatureGroup,
)
from mloda_plugins.feature_group.experimental.environment.list_directory_feature_group import (
    ListDirectoryFeatureGroup,
)
from mloda_plugins.feature_group.experimental.source_input_feature import SourceInputFeature


def make_toy_group() -> type[FeatureGroup]:
    class ToyGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            "mode": PropertySpec("Required mode.", allowed_values=("fast", "slow"), strict_validation=True),
            "scale": PropertySpec("Optional scale.", default="1"),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"toy_a"}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature("toy_src")}

    return ToyGroup


def make_always_matches() -> type[FeatureGroup]:
    class AlwaysMatches(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"am_a"}

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return True

    return AlwaysMatches


def make_rejects_declared_name() -> type[FeatureGroup]:
    class RejectsDeclaredName(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"rd_a", "rd_b"}

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) in {"rd_a", cls.get_class_name()}

    return RejectsDeclaredName


def make_self_input() -> type[FeatureGroup]:
    class SelfInput(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"self_in"}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature("self_in")}

    return SelfInput


def make_ignores_strict_values() -> type[FeatureGroup]:
    class IgnoresStrictValues(FeatureGroup):
        PROPERTY_MAPPING = {
            "mode": PropertySpec("Mode.", allowed_values=("a", "b"), strict_validation=True, default="a"),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"is_a"}

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) in {"is_a", cls.get_class_name()}

    return IgnoresStrictValues


def test_toy_group_passes() -> None:
    result = check_feature_group_shape(make_toy_group())
    assert result.check == "feature_group_shape"
    assert result.subject == "ToyGroup"
    assert result.failures == ()


def test_toy_group_passes_with_explicit_base_options() -> None:
    result = check_feature_group_shape(make_toy_group(), base_options=Options(group={"mode": "slow"}))
    assert result.passed


@pytest.mark.parametrize("cls", [InstalledPackagesFeatureGroup, ListDirectoryFeatureGroup, SourceInputFeature])
def test_builtin_groups_pass(cls: type[FeatureGroup]) -> None:
    assert check_feature_group_shape(cls).failures == ()


@pytest.mark.parametrize(
    ("make", "prefix"),
    [
        (make_always_matches, "foreign:"),
        (make_rejects_declared_name, "names:"),
        (make_self_input, "inputs:"),
        (make_ignores_strict_values, "strict:"),
    ],
)
def test_broken_group_fails_only_its_sub_check(make: Callable[[], type[FeatureGroup]], prefix: str) -> None:
    result = check_feature_group_shape(make())
    assert not result.passed
    assert all(f.startswith(prefix) for f in result.failures), result.failures


def test_unbuildable_request_is_reported_not_raised() -> None:
    class NoDerivableKey(FeatureGroup):
        PROPERTY_MAPPING = {"free": PropertySpec("Free-form required key.")}

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"nd_a"}

    result = check_feature_group_shape(NoDerivableKey)
    assert any(f.startswith("names:") and "cannot build a matching request" in f for f in result.failures)


def test_assert_passes_silently_for_conforming_group() -> None:
    assert_feature_group_shape(make_toy_group())


def test_assert_raises_pytest_failure_for_broken_group() -> None:
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_feature_group_shape(make_always_matches())
    assert "foreign:" in str(excinfo.value)
