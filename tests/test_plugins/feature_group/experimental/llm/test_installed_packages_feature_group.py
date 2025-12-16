from typing import List
from mloda import Feature
from mloda.provider import FeatureSet
from mloda_plugins.feature_group.experimental.llm.installed_packages_feature_group import InstalledPackagesFeatureGroup
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
import mloda


def test_installed_packages_feature_group() -> None:
    feature_set = FeatureSet()
    result = InstalledPackagesFeatureGroup.calculate_feature(None, feature_set)
    assert InstalledPackagesFeatureGroup.get_class_name() in result
    assert isinstance(result[InstalledPackagesFeatureGroup.get_class_name()], list)


def test_installed_packages_feature_group_mlodaAPI() -> None:
    features: List[Feature | str] = [InstalledPackagesFeatureGroup.get_class_name()]
    result = mloda.run_all(features, compute_frameworks={PandasDataFrame})
    assert len(result) == 1
    assert InstalledPackagesFeatureGroup.get_class_name() in result[0]
