from pathlib import PosixPath
from typing import List
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.feature_group.experimental.llm.list_directory_feature_group import ListDirectoryFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_core.api.request import mlodaAPI


def test_list_directory_feature_group(tmp_path: PosixPath) -> None:
    # Create a dummy directory and file within the temporary test path
    test_dir = tmp_path / "test_dir"
    test_file = test_dir / "test_file.txt"

    test_dir.mkdir()
    test_file.write_text("test content")

    feature_set = FeatureSet()

    result = ListDirectoryFeatureGroup.calculate_feature(None, feature_set)

    assert ListDirectoryFeatureGroup.get_class_name() in result
    assert isinstance(result[ListDirectoryFeatureGroup.get_class_name()], list)
    assert "mloda_core" in result[ListDirectoryFeatureGroup.get_class_name()][0]


def test_list_directory_feature_group_mlodaAPI() -> None:
    # This test checks if ListDirectoryFeatureGroup can be run via mlodaAPI
    features: List[Feature | str] = [ListDirectoryFeatureGroup.get_class_name()]
    result = mlodaAPI.run_all(features, compute_frameworks={PandasDataframe})
    assert len(result) == 1
    assert ListDirectoryFeatureGroup.get_class_name() in result[0]
