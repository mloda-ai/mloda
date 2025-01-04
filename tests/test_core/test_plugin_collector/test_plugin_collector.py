import os
from typing import Any
import unittest

import pytest
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from tests.test_core.test_input_data.test_input_data import InputDataTestFeatureGroup


class ATestFeatureGroup(AbstractFeatureGroup):
    pass


class BTestFeatureGroup(AbstractFeatureGroup):
    pass


class TestPlugInCollector(unittest.TestCase):
    def setUp(self) -> None:
        self.plugin_collector = PlugInCollector()

    def test_add_disabled_feature_group_classes(self) -> None:
        self.plugin_collector.add_disabled_feature_group_classes({ATestFeatureGroup})
        self.assertIn(ATestFeatureGroup, self.plugin_collector.disabled_feature_group_classes)

    def test_add_enabled_feature_group_classes(self) -> None:
        self.plugin_collector.add_enabled_feature_group_classes({BTestFeatureGroup})
        self.assertIn(BTestFeatureGroup, self.plugin_collector.enabled_feature_group_classes)

    def test_applicable_feature_group_class_disabled(self) -> None:
        self.plugin_collector.add_disabled_feature_group_classes({ATestFeatureGroup})
        self.assertFalse(self.plugin_collector.applicable_feature_group_class(ATestFeatureGroup))

    def test_applicable_feature_group_class_enabled(self) -> None:
        self.plugin_collector.add_enabled_feature_group_classes({BTestFeatureGroup})
        self.assertTrue(self.plugin_collector.applicable_feature_group_class(BTestFeatureGroup))

    def test_applicable_feature_group_class_no_enabled(self) -> None:
        self.assertTrue(self.plugin_collector.applicable_feature_group_class(ATestFeatureGroup))

    def test_disabled_feature_groups_static_method(self) -> None:
        plugin_collector = PlugInCollector.disabled_feature_groups({ATestFeatureGroup})
        self.assertIn(ATestFeatureGroup, plugin_collector.disabled_feature_group_classes)

    def test_enabled_feature_groups_static_method(self) -> None:
        plugin_collector = PlugInCollector.enabled_feature_groups({BTestFeatureGroup})
        self.assertIn(BTestFeatureGroup, plugin_collector.enabled_feature_group_classes)


class TestPlugInCollectorIntegration:
    file_path = f"{os.path.dirname(os.path.abspath(__file__))}/creditcard_2023.csv"
    feature_names = "id,V1,V2"
    feature_list = feature_names.split(",")

    def test_enabled_plugins(self) -> Any:
        features = [f"InputDataTestFeatureGroup_{f}" for f in self.feature_list]
        mlodaAPI.run_all(
            features,  # type: ignore
            compute_frameworks=["PyarrowTable"],
            plugin_collector=PlugInCollector.enabled_feature_groups({InputDataTestFeatureGroup}),
        )

        with pytest.raises(ValueError):
            mlodaAPI.run_all(
                features,  # type: ignore
                compute_frameworks=["PyarrowTable"],
                plugin_collector=PlugInCollector.enabled_feature_groups({BTestFeatureGroup}),
            )

    def test_disabled_plugins(self) -> Any:
        features = [f"InputDataTestFeatureGroup_{f}" for f in self.feature_list]

        with pytest.raises(ValueError):
            mlodaAPI.run_all(
                features,  # type: ignore
                compute_frameworks=["PyarrowTable"],
                plugin_collector=PlugInCollector.disabled_feature_groups({InputDataTestFeatureGroup}),
            )

        mlodaAPI.run_all(
            features,  # type: ignore
            compute_frameworks=["PyarrowTable"],
            plugin_collector=PlugInCollector.disabled_feature_groups({BTestFeatureGroup}),
        )
