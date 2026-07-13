"""Tests for issue #692: abstract FeatureGroup bases must not break matching.

The default `match_feature_group_criteria` calls `_is_root_and_matches_input_data`,
which instantiates `cls()`. For an abstract base (an @abstractmethod, no matcher
override, no FeatureChainParserMixin) this raises TypeError for every feature name
and aborts resolution for the whole run. Such a base must simply not match.
"""

import inspect
from abc import abstractmethod
from typing import Any, Optional

import pandas as pd

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import BaseInputData, DataCreator
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class AbstractBaseWithAbstractMethod(FeatureGroup):
    """Abstract base: cannot be instantiated, does not override the matcher."""

    @classmethod
    @abstractmethod
    def transform(cls, data: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class ConcreteRootFeatureGroup(FeatureGroup):
    """Concrete root feature group: no input features, creates its own data."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"issue692_root_feature"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"issue692_root_feature": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class ConcreteSubclassOfAbstractBase(AbstractBaseWithAbstractMethod):
    """Leaf that implements the abstract method: instantiable, must match its root feature."""

    @classmethod
    def transform(cls, data: Any) -> Any:
        return data

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"issue692_subclass_root_feature"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"issue692_subclass_root_feature": [4, 5, 6]})


class TestAbstractBaseMatcher:
    def test_abstract_base_does_not_match(self) -> None:
        """The abstract base must return False, not raise TypeError."""
        assert (
            AbstractBaseWithAbstractMethod.match_feature_group_criteria("any_unrelated_feature_name", Options(), None)
            is False
        )

    def test_abstract_base_does_not_match_feature_name_object(self) -> None:
        assert (
            AbstractBaseWithAbstractMethod.match_feature_group_criteria(
                FeatureName("any_unrelated_feature_name"), Options(), None
            )
            is False
        )

    def test_concrete_root_still_matches(self) -> None:
        """Rootness semantics stay unchanged for concrete root feature groups."""
        assert ConcreteRootFeatureGroup.match_feature_group_criteria("issue692_root_feature", Options(), None) is True
        assert ConcreteRootFeatureGroup.match_feature_group_criteria("unrelated_name", Options(), None) is False

    def test_concrete_subclass_of_abstract_base_matches(self) -> None:
        """The check must look at the leaf class, not at the abstractness of its base."""
        assert inspect.isabstract(AbstractBaseWithAbstractMethod) is True
        assert inspect.isabstract(ConcreteSubclassOfAbstractBase) is False
        assert (
            ConcreteSubclassOfAbstractBase.match_feature_group_criteria(
                "issue692_subclass_root_feature", Options(), None
            )
            is True
        )

    def test_non_abstract_feature_group_still_matches(self) -> None:
        """Guard against strengthening the check to issubclass(cls, ABC)."""
        assert inspect.isabstract(ConcreteRootFeatureGroup) is False
        assert ConcreteRootFeatureGroup.match_feature_group_criteria("issue692_root_feature", Options(), None) is True

    def test_abstract_base_coexists_with_unrelated_features_in_run(self) -> None:
        """An abstract base in the enabled set must not break an unrelated run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {AbstractBaseWithAbstractMethod, ConcreteRootFeatureGroup}
        )

        results = mloda.run_all(
            features=[Feature("issue692_root_feature")],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1
        assert list(results[0]["issue692_root_feature"]) == [1, 2, 3]
