from pathlib import PosixPath
from mloda.user import Feature
from mloda_plugins.feature_group.experimental.environment.list_directory_feature_group import (
    ListDirectoryFeatureGroup,
)
from mloda.provider import FeatureSet
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.user import mloda


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
    # This test checks if ListDirectoryFeatureGroup can be run via mloda
    features: list[Feature | str] = [ListDirectoryFeatureGroup.get_class_name()]
    result = mloda.run_all(features, compute_frameworks={PandasDataFrame})
    for res in result:
        assert "__init__.py" not in res[ListDirectoryFeatureGroup.get_class_name()].values[0]
    assert len(result) == 1
    assert ListDirectoryFeatureGroup.get_class_name() in result[0]


def test_is_ignored_directory_pattern_respects_segment_boundary() -> None:
    # A directory pattern must match the directory itself and its contents,
    # but not unrelated paths that merely share a prefix.
    assert ListDirectoryFeatureGroup._is_ignored("build_tools/x.py", {"build/"}) is False
    assert ListDirectoryFeatureGroup._is_ignored("build", {"build/"}) is True
    assert ListDirectoryFeatureGroup._is_ignored("build/out.txt", {"build/"}) is True
    assert ListDirectoryFeatureGroup._is_ignored("build/sub/out.txt", {"build/"}) is True


def test_is_ignored_wildcard_patterns() -> None:
    assert ListDirectoryFeatureGroup._is_ignored("debug.log", {"*.log"}) is True
    assert ListDirectoryFeatureGroup._is_ignored("debug.txt", {"*.log"}) is False
    assert ListDirectoryFeatureGroup._is_ignored("temp.txt", {"temp*"}) is True
    assert ListDirectoryFeatureGroup._is_ignored("keep.txt", {"temp*"}) is False
    assert ListDirectoryFeatureGroup._is_ignored("logs/app.log", {"logs/*"}) is True
    assert ListDirectoryFeatureGroup._is_ignored("other/app.log", {"logs/*"}) is False
