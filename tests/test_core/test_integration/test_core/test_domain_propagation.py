"""Integration tests for domain propagation through feature chains.

These tests verify that domain propagation works end-to-end through the full
mloda execution pipeline, from parent features down to child features in the
dependency chain.

Uses pytest.mark.parametrize for matrix testing of various scenarios.
"""

from typing import Any, Optional, Set, Type, Union

import pytest

from mloda.user import Feature, Features, PluginCollector, Domain
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import DataCreator, BaseInputData
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


TEST_DOMAIN = Domain("TestDomain")


class DomainTestDataCreator(FeatureGroup):
    """Data creator that provides test data for domain propagation tests."""

    @classmethod
    def get_domain(cls) -> Domain:
        return TEST_DOMAIN

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"base_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"base_value": [10, 20, 30, 40, 50]})

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class ChildFeatureGroup(FeatureGroup):
    """Child feature group that depends on base_value."""

    @classmethod
    def get_domain(cls) -> Domain:
        return TEST_DOMAIN

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "child_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {"base_value"}  # type: ignore[arg-type]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["child_feature"] = data["base_value"] * 2
        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class ParentFeatureGroup(FeatureGroup):
    """Parent feature group that depends on ChildFeatureGroup."""

    @classmethod
    def get_domain(cls) -> Domain:
        return TEST_DOMAIN

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "parent_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {"child_feature"}  # type: ignore[arg-type]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["parent_feature"] = data["child_feature"] + 100
        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class GrandchildFeatureGroup(FeatureGroup):
    """Grandchild feature group at the bottom of a 3-level chain."""

    @classmethod
    def get_domain(cls) -> Domain:
        return TEST_DOMAIN

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "grandchild_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {"base_value"}  # type: ignore[arg-type]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["grandchild_feature"] = data["base_value"] + 1
        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class IntermediateFeatureGroup(FeatureGroup):
    """Intermediate feature group in a 3-level chain."""

    @classmethod
    def get_domain(cls) -> Domain:
        return TEST_DOMAIN

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "intermediate_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {"grandchild_feature"}  # type: ignore[arg-type]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["intermediate_feature"] = data["grandchild_feature"] * 10
        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class GrandparentFeatureGroup(FeatureGroup):
    """Top-level feature group in a 3-level chain."""

    @classmethod
    def get_domain(cls) -> Domain:
        return TEST_DOMAIN

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "grandparent_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {"intermediate_feature"}  # type: ignore[arg-type]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["grandparent_feature"] = data["intermediate_feature"] + 1000
        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


PLUGIN_COLLECTOR = PluginCollector.enabled_feature_groups(
    {
        DomainTestDataCreator,
        ChildFeatureGroup,
        ParentFeatureGroup,
        GrandchildFeatureGroup,
        IntermediateFeatureGroup,
        GrandparentFeatureGroup,
    }
)


class TestDomainPropagationMatrix:
    """Matrix tests for domain propagation through feature chains."""

    @pytest.mark.parametrize(
        "parent_feature_name,parent_domain,expected_children_with_domain",
        [
            # 2-level chain: parent -> child -> base_value
            ("parent_feature", "TestDomain", {"child_feature", "base_value"}),
            # 3-level chain: grandparent -> intermediate -> grandchild -> base_value
            ("grandparent_feature", "TestDomain", {"intermediate_feature", "grandchild_feature", "base_value"}),
        ],
        ids=["2-level-chain", "3-level-chain"],
    )
    def test_domain_propagates_through_chain(
        self,
        parent_feature_name: str,
        parent_domain: str,
        expected_children_with_domain: Set[str],
    ) -> None:
        """Domain should propagate from parent to all children in the chain."""
        from mloda.core.core.engine import Engine

        parent_feature = Feature(parent_feature_name, domain=parent_domain)
        features = Features([parent_feature])

        engine = Engine(
            features,
            {PandasDataFrame},
            None,
            plugin_collector=PLUGIN_COLLECTOR,
        )

        found_features: Set[str] = set()
        for feature_group_class, feature_set in engine.feature_group_collection.items():
            for feature in feature_set:
                if feature.name.name in expected_children_with_domain:
                    assert feature.domain is not None, f"{feature.name.name} should have inherited domain but has None"
                    assert feature.domain.name == parent_domain, (
                        f"{feature.name.name} should have domain '{parent_domain}' but has '{feature.domain.name}'"
                    )
                    found_features.add(feature.name.name)

        assert found_features == expected_children_with_domain, (
            f"Expected features {expected_children_with_domain} but found {found_features}"
        )

    @pytest.mark.parametrize(
        "parent_feature_name,child_to_check",
        [
            ("parent_feature", "child_feature"),
            ("grandparent_feature", "intermediate_feature"),
        ],
        ids=["2-level-no-domain", "3-level-no-domain"],
    )
    def test_no_domain_propagation_when_parent_has_no_domain(
        self,
        parent_feature_name: str,
        child_to_check: str,
    ) -> None:
        """When parent has no domain, children should have no domain."""
        from mloda.core.core.engine import Engine

        parent_feature = Feature(parent_feature_name)  # No domain
        features = Features([parent_feature])

        engine = Engine(
            features,
            {PandasDataFrame},
            None,
            plugin_collector=PLUGIN_COLLECTOR,
        )

        for feature_group_class, feature_set in engine.feature_group_collection.items():
            for feature in feature_set:
                if feature.name.name == child_to_check:
                    assert feature.domain is None, f"{child_to_check} should have no domain but has '{feature.domain}'"

    @pytest.mark.parametrize(
        "child_input,parent_domain,expected_child_domain",
        [
            # String feature inherits parent domain
            ("child_feature", "TestDomain", "TestDomain"),
            # Feature without domain inherits parent domain
            (Feature("child_feature"), "TestDomain", "TestDomain"),
            # Feature with explicit domain keeps its domain
            (Feature("child_feature", domain="ChildDomain"), "ParentDomain", "ChildDomain"),
            # String feature with no parent domain has no domain
            ("child_feature", None, None),
        ],
        ids=[
            "string-inherits-domain",
            "feature-no-domain-inherits",
            "feature-explicit-domain-preserved",
            "string-no-parent-domain",
        ],
    )
    def test_features_class_domain_propagation(
        self,
        child_input: Union[str, Feature],
        parent_domain: Optional[str],
        expected_child_domain: Optional[str],
    ) -> None:
        """Test Features class handles domain propagation correctly."""
        features = Features([child_input], parent_domain=parent_domain)

        child_feature = features.collection[0]
        if expected_child_domain is None:
            assert child_feature.domain is None, f"Expected no domain but got '{child_feature.domain}'"
        else:
            assert child_feature.domain is not None, f"Expected domain '{expected_child_domain}' but got None"
            assert child_feature.domain.name == expected_child_domain, (
                f"Expected domain '{expected_child_domain}' but got '{child_feature.domain.name}'"
            )
