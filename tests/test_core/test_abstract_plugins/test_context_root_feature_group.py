"""Red-phase tests for ContextRootFeatureGroup convenience base class.

These pin the contract of the not-yet-existing ``ContextRootFeatureGroup`` (a
``FeatureGroup`` subclass for root feature groups that generate data in place via
``DataCreator``). They MUST fail until the class exists and is exported from
``mloda.provider``. Test FeatureGroup subclasses use the ``crfg_`` prefix so the
globally-discovered feature names stay unique and parallel-safe under xdist.
"""

import inspect
from typing import Any, ClassVar, Optional

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import ContextRootFeatureGroup
from mloda.provider import DataCreator
from mloda.provider import FeatureSet
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


def test_import_from_provider() -> None:
    # The class must be importable from the public provider namespace.
    assert ContextRootFeatureGroup is not None


def test_feature_names_supported_returns_copy() -> None:
    class _CrfgNames(ContextRootFeatureGroup):
        FEATURES: ClassVar[set[str]] = {"crfg_a", "crfg_b"}

    assert _CrfgNames.feature_names_supported() == {"crfg_a", "crfg_b"}

    # Mutating the returned set must not affect the class attribute.
    returned = _CrfgNames.feature_names_supported()
    returned.add("crfg_mutated")
    assert _CrfgNames.FEATURES == {"crfg_a", "crfg_b"}


def test_input_data_is_data_creator_with_features() -> None:
    class _CrfgInput(ContextRootFeatureGroup):
        FEATURES: ClassVar[set[str]] = {"crfg_a", "crfg_b"}

    input_data = _CrfgInput.input_data()
    assert isinstance(input_data, DataCreator)
    assert input_data.feature_names == {"crfg_a", "crfg_b"}


def test_match_feature_group_criteria_base_and_sub_column() -> None:
    class _CrfgMatch(ContextRootFeatureGroup):
        FEATURES: ClassVar[set[str]] = {"crfg_a", "crfg_b"}

    options = Options()
    assert _CrfgMatch.match_feature_group_criteria("crfg_a", options) is True
    # Sub-column request (~N suffix) resolves to the base feature name.
    assert _CrfgMatch.match_feature_group_criteria("crfg_a~0", options) is True
    assert _CrfgMatch.match_feature_group_criteria("not_declared", options) is False


def test_compute_framework_rule_default_none() -> None:
    class _CrfgDefaultCf(ContextRootFeatureGroup):
        FEATURES: ClassVar[set[str]] = {"crfg_default_cf"}

    assert _CrfgDefaultCf.compute_framework_rule() is None


def test_compute_framework_rule_declared() -> None:
    class _CrfgDeclaredCf(ContextRootFeatureGroup):
        FEATURES: ClassVar[set[str]] = {"crfg_declared_cf"}
        COMPUTE_FRAMEWORKS: ClassVar[Optional[set[type]]] = {PythonDictFramework}

    assert _CrfgDeclaredCf.compute_framework_rule() == {PythonDictFramework}


class _CrfgRoot(ContextRootFeatureGroup):
    FEATURES: ClassVar[set[str]] = {"crfg_x", "crfg_y"}
    COMPUTE_FRAMEWORKS: ClassVar[Optional[set[type]]] = {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"crfg_x": [1, 2, 3], "crfg_y": [4, 5, 6]}


def test_end_to_end_run_all() -> None:
    result = mloda.run_all(["crfg_x", "crfg_y"], compute_frameworks=["PythonDictFramework"])
    assert len(result) == 1
    assert result[0] == {"crfg_x": [1, 2, 3], "crfg_y": [4, 5, 6]}


def test_match_feature_group_criteria_override() -> None:
    class _CrfgOverride(ContextRootFeatureGroup):
        FEATURES: ClassVar[set[str]] = {"crfg_declared_only"}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: Any,
            options: Options,
            data_access_collection: Any = None,
        ) -> bool:
            return str(feature_name).startswith("custom_")

    options = Options()
    # Override takes effect: prefix match, not FEATURES membership.
    assert _CrfgOverride.match_feature_group_criteria("custom_thing", options) is True
    assert _CrfgOverride.match_feature_group_criteria("crfg_declared_only", options) is False


def test_base_class_is_abstract() -> None:
    # The base must be abstract so it is not treated as a concrete shippable plugin.
    assert inspect.isabstract(ContextRootFeatureGroup) is True


def test_subclass_without_calculate_feature_is_abstract() -> None:
    class _CrfgNoCalc(ContextRootFeatureGroup):
        FEATURES: ClassVar[set[str]] = {"crfg_nocalc"}

    # Declaring only FEATURES leaves calculate_feature abstract: not instantiable.
    assert inspect.isabstract(_CrfgNoCalc) is True
    with pytest.raises(TypeError):
        _CrfgNoCalc()  # type: ignore[abstract]

    class _CrfgWithCalc(ContextRootFeatureGroup):
        FEATURES: ClassVar[set[str]] = {"crfg_withcalc"}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {"crfg_withcalc": [1]}

    # Defining calculate_feature makes the subclass concrete and instantiable.
    assert inspect.isabstract(_CrfgWithCalc) is False
    assert _CrfgWithCalc() is not None


def test_compute_framework_rule_returns_copy() -> None:
    class _CrfgCfCopy(ContextRootFeatureGroup):
        FEATURES: ClassVar[set[str]] = {"crfg_cf_copy"}
        COMPUTE_FRAMEWORKS: ClassVar[Optional[set[type]]] = {PythonDictFramework}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {"crfg_cf_copy": [1]}

    rule = _CrfgCfCopy.compute_framework_rule()
    assert rule == {PythonDictFramework}

    # Mutating the returned rule must not affect the class attribute (defensive copy).
    rule.clear()
    assert _CrfgCfCopy.COMPUTE_FRAMEWORKS == {PythonDictFramework}
