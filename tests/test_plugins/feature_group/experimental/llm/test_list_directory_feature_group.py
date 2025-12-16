from pathlib import PosixPath
from typing import List
from mloda import Feature
from mloda_plugins.feature_group.experimental.llm.list_directory_feature_group import ListDirectoryFeatureGroup
from mloda.provider import FeatureSet
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
import mloda


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
    assert "mloda" in result[ListDirectoryFeatureGroup.get_class_name()][0]


def test_list_directory_feature_group_mlodaAPI() -> None:
    # This test checks if ListDirectoryFeatureGroup can be run via API
    features: List[Feature | str] = [ListDirectoryFeatureGroup.get_class_name()]
    result = mloda.run_all(features, compute_frameworks={PandasDataFrame})
    for res in result:
        assert "__init__.py" not in res[ListDirectoryFeatureGroup.get_class_name()].values[0]
    assert len(result) == 1
    assert ListDirectoryFeatureGroup.get_class_name() in result[0]
