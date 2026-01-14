import os
from typing import Any

import pytest
from mloda.provider import FeatureGroup
from mloda.user import PluginCollector
from mloda.user import mloda
from tests.test_plugins.feature_group.input_data.test_input_data import InputDataTestFeatureGroup


class ATestFeatureGroup(FeatureGroup):
    pass


class BTestFeatureGroup(FeatureGroup):
    pass


class TestPluginCollector:
    def setup_method(self) -> None:
        self.plugin_collector = PluginCollector()

    def test_add_disabled_feature_group_classes(self) -> None:
        self.plugin_collector.add_disabled_feature_group_classes({ATestFeatureGroup})
        assert ATestFeatureGroup in self.plugin_collector.disabled_feature_group_classes

    def test_add_enabled_feature_group_classes(self) -> None:
        self.plugin_collector.add_enabled_feature_group_classes({BTestFeatureGroup})
        assert BTestFeatureGroup in self.plugin_collector.enabled_feature_group_classes

    def test_applicable_feature_group_class_disabled(self) -> None:
        self.plugin_collector.add_disabled_feature_group_classes({ATestFeatureGroup})
        assert not self.plugin_collector.applicable_feature_group_class(ATestFeatureGroup)

    def test_applicable_feature_group_class_enabled(self) -> None:
        self.plugin_collector.add_enabled_feature_group_classes({BTestFeatureGroup})
        assert self.plugin_collector.applicable_feature_group_class(BTestFeatureGroup)

    def test_applicable_feature_group_class_no_enabled(self) -> None:
        assert self.plugin_collector.applicable_feature_group_class(ATestFeatureGroup)

    def test_disabled_feature_groups_static_method(self) -> None:
        plugin_collector = PluginCollector.disabled_feature_groups({ATestFeatureGroup})
        assert ATestFeatureGroup in plugin_collector.disabled_feature_group_classes

    def test_enabled_feature_groups_static_method(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({BTestFeatureGroup})
        assert BTestFeatureGroup in plugin_collector.enabled_feature_group_classes


class TestPluginCollectorIntegration:
    file_path = f"{os.path.dirname(os.path.abspath(__file__))}/creditcard_2023.csv"
    feature_names = "id,V1,V2"
    feature_list = feature_names.split(",")

    def test_enabled_plugins(self) -> Any:
        features = [f"InputDataTestFeatureGroup_{f}" for f in self.feature_list]
        mloda.run_all(
            features,  # type: ignore
            compute_frameworks=["PyArrowTable"],
            plugin_collector=PluginCollector.enabled_feature_groups({InputDataTestFeatureGroup}),
        )

        with pytest.raises(ValueError):
            mloda.run_all(
                features,  # type: ignore
                compute_frameworks=["PyArrowTable"],
                plugin_collector=PluginCollector.enabled_feature_groups({BTestFeatureGroup}),
            )

    def test_disabled_plugins(self) -> Any:
        features = [f"InputDataTestFeatureGroup_{f}" for f in self.feature_list]

        with pytest.raises(ValueError):
            mloda.run_all(
                features,  # type: ignore
                compute_frameworks=["PyArrowTable"],
                plugin_collector=PluginCollector.disabled_feature_groups({InputDataTestFeatureGroup}),
            )

        mloda.run_all(
            features,  # type: ignore
            compute_frameworks=["PyArrowTable"],
            plugin_collector=PluginCollector.disabled_feature_groups({BTestFeatureGroup}),
        )
