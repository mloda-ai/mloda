"""Red-phase tests for ContextRootFeatureGroup convenience base class.

These pin the contract of the not-yet-existing ``ContextRootFeatureGroup`` (a
``FeatureGroup`` subclass for root feature groups that generate data in place via
``DataCreator``). They MUST fail until the class exists and is exported from
``mloda.provider``. Test FeatureGroup subclasses use the ``crfg_`` prefix so the
globally-discovered feature names stay unique and parallel-safe under xdist.
"""

from typing import Any, ClassVar, Optional

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
