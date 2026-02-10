"""Tests for class identity based on actual class objects rather than names.

Issue #184: Domain-Based Class Grouping Problem

These tests verify that FeatureGroup uses actual class identity (via `type(self) is type(another)`)
rather than name-based identity. This ensures that dynamically created classes with the same
`__name__` but different domains (or different class objects) are treated as distinct entities.
"""

from typing import Any, Optional, Set, Type, Union

import pandas as pd
import pytest

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DataCreator, BaseInputData
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


def create_feature_group_class(name: str, domain: Domain) -> type:
    """Factory function to create FeatureGroup subclasses with specific name and domain."""

    class DynamicFeatureGroup(FeatureGroup):
        @classmethod
        def get_domain(cls) -> Domain:
            return domain

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Any]]:
            return None

    DynamicFeatureGroup.__name__ = name
    DynamicFeatureGroup.__qualname__ = name
    return DynamicFeatureGroup


class TestFeatureGroupClassIdentity:
    """Tests for FeatureGroup class identity based on domain."""

    def test_same_name_different_domain_feature_groups_are_distinct(self) -> None:
        """Two FeatureGroup subclasses with same __name__ but different domains should not be equal.

        Creates two dynamically generated FeatureGroup subclasses with __name__ = "DynamicFeature"
        but different domains ("Sales" vs "Finance"). They should be treated as distinct.
        """
        sales_domain = Domain("Sales")
        finance_domain = Domain("Finance")

        SalesFeatureGroup = create_feature_group_class("DynamicFeature", sales_domain)
        FinanceFeatureGroup = create_feature_group_class("DynamicFeature", finance_domain)

        sales_instance = SalesFeatureGroup()
        finance_instance = FinanceFeatureGroup()

        assert sales_instance != finance_instance, (
            "FeatureGroups with same name but different domains should not be equal"
        )
        assert hash(sales_instance) != hash(finance_instance), (
            "FeatureGroups with same name but different domains should have different hashes"
        )

    def test_same_name_different_domain_in_set(self) -> None:
        """Both classes with same name but different domains should survive in a set.

        FeatureGroup instances from different domains should be kept separate in sets
        even if their class names are identical.
        """
        sales_domain = Domain("Sales")
        finance_domain = Domain("Finance")

        SalesFeatureGroup = create_feature_group_class("DynamicFeature", sales_domain)
        FinanceFeatureGroup = create_feature_group_class("DynamicFeature", finance_domain)

        sales_instance = SalesFeatureGroup()
        finance_instance = FinanceFeatureGroup()

        feature_group_set = {sales_instance, finance_instance}

        assert len(feature_group_set) == 2, (
            f"Set should contain 2 distinct FeatureGroups but contains {len(feature_group_set)}. "
            "This indicates that domain-differentiated classes are incorrectly treated as equal."
        )

    def test_same_name_same_domain_different_classes_are_distinct(self) -> None:
        """Different classes with same name AND domain should still be distinct.

        Even when two classes share the same name and domain, if they were created
        by separate factory calls (different class objects), they should be distinct.
        """
        sales_domain = Domain("Sales")

        FeatureGroupA = create_feature_group_class("DynamicFeature", sales_domain)
        FeatureGroupB = create_feature_group_class("DynamicFeature", sales_domain)

        assert FeatureGroupA is not FeatureGroupB, "Factory should create distinct class objects"

        instance_a = FeatureGroupA()
        instance_b = FeatureGroupB()

        assert instance_a != instance_b, (
            "Instances from different class objects should not be equal, even if they have the same name and domain"
        )
        assert hash(instance_a) != hash(instance_b), (
            "Instances from different class objects should have different hashes"
        )

    def test_same_class_instances_are_equal(self) -> None:
        """Multiple instances of the same class should be equal.

        This is a sanity check to ensure the basic equality contract is maintained.
        Two instances of the exact same class should be equal.
        """
        sales_domain = Domain("Sales")

        SalesFeatureGroup = create_feature_group_class("SalesFeature", sales_domain)

        instance1 = SalesFeatureGroup()
        instance2 = SalesFeatureGroup()

        assert instance1 == instance2, "Two instances of the same class should be equal"
        assert hash(instance1) == hash(instance2), "Two instances of the same class should have the same hash"

    def test_feature_groups_in_dict_preserve_domain_distinction(self) -> None:
        """FeatureGroups used as dict keys should preserve domain distinction.

        FeatureGroup instances from different domains should coexist as separate dict keys.
        """
        sales_domain = Domain("Sales")
        finance_domain = Domain("Finance")

        SalesFeatureGroup = create_feature_group_class("DynamicFeature", sales_domain)
        FinanceFeatureGroup = create_feature_group_class("DynamicFeature", finance_domain)

        sales_instance = SalesFeatureGroup()
        finance_instance = FinanceFeatureGroup()

        feature_dict = {
            sales_instance: "sales_data",
            finance_instance: "finance_data",
        }

        assert len(feature_dict) == 2, (
            f"Dict should have 2 entries but has {len(feature_dict)}. "
            "Domain-differentiated FeatureGroups should be distinct dict keys."
        )
        assert feature_dict[sales_instance] == "sales_data"
        assert feature_dict[finance_instance] == "finance_data"


def create_domain_feature_group(domain_name: str, feature_value: int) -> Type[FeatureGroup]:
    """Create a FeatureGroup class with a specific domain and feature value.

    All created classes have __name__ = "DomainHandler" to test class identity.
    """
    domain = Domain(domain_name)

    class DomainHandler(FeatureGroup):
        @classmethod
        def get_domain(cls) -> Domain:
            return domain

        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return DataCreator({"domain_feature"})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return pd.DataFrame({"domain_feature": [feature_value]})

        @classmethod
        def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
            return {PandasDataFrame}

    return DomainHandler


class TestClassIdentityIntegration:
    """Integration tests proving class identity works with mloda.run_all."""

    def test_same_name_different_domain_run_all_returns_separate_results(self) -> None:
        """Two FeatureGroups with same __name__ but different domains produce separate results.

        Integration test for Issue #184: two dynamically created classes with
        __name__ = "DomainHandler" but different domains ("Sales" vs "Finance")
        are treated as distinct by mloda.run_all and produce separate result dataframes.
        """
        SalesHandler = create_domain_feature_group("Sales", feature_value=100)
        FinanceHandler = create_domain_feature_group("Finance", feature_value=200)

        assert SalesHandler.__name__ == FinanceHandler.__name__ == "DomainHandler"

        plugin_collector = PluginCollector.enabled_feature_groups({SalesHandler, FinanceHandler})

        sales_feature = Feature("domain_feature", domain="Sales")
        finance_feature = Feature("domain_feature", domain="Finance")

        results = mloda.run_all(
            features=[sales_feature, finance_feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 2, (
            f"Expected 2 separate results (one per domain) but got {len(results)}. "
            "This indicates same-name classes are being conflated."
        )

        values = sorted([int(r["domain_feature"].iloc[0]) for r in results])
        assert values == [100, 200], (
            f"Expected values [100, 200] from Sales and Finance domains but got {values}. "
            "This indicates domain-specific calculations are not being preserved."
        )
