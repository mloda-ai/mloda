import subprocess
from unittest.mock import patch

from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda_plugins.feature_group.experimental.environment.installed_packages_feature_group import (
    InstalledPackagesFeatureGroup,
)
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.user import mloda


def test_installed_packages_feature_group() -> None:
    feature_set = FeatureSet()
    result = InstalledPackagesFeatureGroup.calculate_feature(None, feature_set)
    assert InstalledPackagesFeatureGroup.get_class_name() in result
    assert isinstance(result[InstalledPackagesFeatureGroup.get_class_name()], list)


def test_installed_packages_feature_group_mlodaAPI() -> None:
    features: list[Feature | str] = [InstalledPackagesFeatureGroup.get_class_name()]
    result = mloda.run_all(features, compute_frameworks={PandasDataFrame})
    assert len(result) == 1
    assert InstalledPackagesFeatureGroup.get_class_name() in result[0]


def test_installed_packages_feature_group_error_path_keeps_the_output_shape() -> None:
    """A pip freeze failure must land in the documented column, not a separate "error" key.

    The success path returns {ClassName: [...]}, and the class docstring's Output
    Format section promises that single column. A different key on failure gives a
    caller a KeyError instead of a readable message, exactly when it is debugging.
    """
    feature_set = FeatureSet()
    failure = subprocess.CalledProcessError(returncode=1, cmd=["pip", "freeze"], stderr="boom")

    with patch("subprocess.run", side_effect=failure):
        result = InstalledPackagesFeatureGroup.calculate_feature(None, feature_set)

    column = InstalledPackagesFeatureGroup.get_class_name()
    assert set(result) == {column}, "the failure path must not introduce another key"
    assert isinstance(result[column], list)
    assert len(result[column]) == 1
    assert "boom" in result[column][0]
    assert "return code 1" in result[column][0]
