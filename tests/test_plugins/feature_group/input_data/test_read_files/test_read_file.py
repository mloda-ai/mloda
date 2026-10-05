import csv
import importlib
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

import pyarrow as pa
import pyarrow.compute as pc

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.provider import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_file_source_transformer import (
    FileSourcePyArrowTransformer,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401
from mloda.user import DataAccessCollection
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec, Link
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from tests.mixins.reader_feature_groups.lazy_format_group import load_group
from tests.test_core.test_integration.test_core.test_runner_one_compute_framework import SumFeature  # noqa: F401

DOUBLING_MARKER = "overwritten_csv_doubling_marker"


@pytest.fixture(autouse=True)
def _stock_formats_loaded() -> None:
    importlib.import_module("mloda_plugins.feature_group.input_data.file_formats.stock_formats")


def _overwritten_group() -> Any:
    """A fresh CsvFG subclass per test (so it is collected afterwards) that doubles every value when marked."""
    csv_group = load_group("csv_fg", "CsvFG")

    class OverwrittenReadCsvInputDataTestFeatureGroup(csv_group):  # type: ignore[misc,valid-type]
        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            # Gated: without the marker this subclass never claims, so it cannot take over other tests' features.
            if options.get(DOUBLING_MARKER) is None:
                return False
            return bool(super().match_feature_group_criteria(feature_name, options, data_access_collection))

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            result = super().calculate_feature(data, features)
            # The stock PyArrowTable loader yields a table; a FileSource descriptor is materialized the same way.
            if isinstance(result, FileSource):
                result = FileSourcePyArrowTransformer.transform_fw_to_other_fw(result)
            new_columns = {
                col_name: pa.array([value * 2 for value in result[col_name].to_pylist()])
                for col_name in result.schema.names
            }
            return pa.table(new_columns)

    return OverwrittenReadCsvInputDataTestFeatureGroup


class TestInputData:
    file_path = f"{os.getcwd()}/tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"

    feature_names = "id,V1,V2"
    feature_list = feature_names.split(",")

    @classmethod
    def get_features(
        cls, features: list[str], path: str | None = None, additional_options: dict[str, Any] = {}
    ) -> list[str | Feature]:
        _feature_list: list[str | Feature] = []
        for feature in features:
            _f = Feature(name=feature)
            for k, v in additional_options.items():
                _f.options.add_to_group(k, v)
            if path is not None:
                _f.options.add_to_group("CsvFG", path)
            _feature_list.append(_f)
        return _feature_list

    def test_local_scope_file(self) -> Any:
        features = self.get_features(self.feature_list, self.file_path)
        result = mloda.run_all(features, compute_frameworks=["PyArrowTable"])
        assert "V2" in result[0].to_pydict()

    def test_local_scope_folder(self) -> Any:
        file_path = self.file_path.replace("creditcard_2023_short.csv", "")
        features = self.get_features(self.feature_list, file_path)
        result = mloda.run_all(features, compute_frameworks=["PyArrowTable"])
        assert "V2" in result[0].to_pydict()

    def test_global_scope_file(self) -> Any:
        result = mloda.run_all(
            self.feature_list,  # type: ignore
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(files={self.file_path}),
        )
        assert "V2" in result[0].to_pydict()

    def test_global_scope_folder(self) -> Any:
        file_path = self.file_path.replace("creditcard_2023_short.csv", "")
        result = mloda.run_all(
            self.feature_list,  # type: ignore
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(folders={file_path}),
        )
        assert "V2" in result[0].to_pydict()

        for k, v in result[0].to_pydict().items():
            if k == "id":
                assert v == [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                ], "We added this to check that overwritting match feature group test was not applied"

    def test_overwriting_match_feature_group_criteria_using_data_access_collection(self) -> Any:
        """
        A gated CsvFG subclass takes over from CsvFG when its marker option is set and it calls the parent.

        Further, this checks if sub_classes are filtered out correctly. (Functionality in IdentifyFeatureGroupClass).
        """
        _group = _overwritten_group()
        features = self.get_features(self.feature_list, None, {DOUBLING_MARKER: "dummy"})

        result = mloda.run_all(
            features,
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(files={self.file_path}),
        )
        assert "V2" in result[0].to_pydict()
        for k, v in result[0].to_pydict().items():
            if k == "id":
                assert v == [0, 2, 4, 6, 8, 10, 12, 14, 16]

    def test_overwriting_match_feature_group_criteria_using_local_scope(self) -> Any:
        """
        A gated CsvFG subclass takes over from CsvFG when its marker option is set and it calls the parent.

        Further, this checks if sub_classes are filtered out correctly. (Functionality in IdentifyFeatureGroupClass).
        """
        _group = _overwritten_group()
        features = self.get_features(self.feature_list, self.file_path, {DOUBLING_MARKER: "dummy"})

        result = mloda.run_all(
            features,
            compute_frameworks=["PyArrowTable"],
        )
        assert "V2" in result[0].to_pydict()
        for k, v in result[0].to_pydict().items():
            if k == "id":
                assert v == [0, 2, 4, 6, 8, 10, 12, 14, 16]

    def test_aggregated_load_csv_with_global_data_access_collection(self) -> Any:
        f = Feature(
            name="sum_of_",
            options={"sum": ("V1", "V2")},
        )
        file_path = self.file_path.replace("creditcard_2023_short.csv", "")
        result = mloda.run_all(
            [f],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(folders={file_path}),
        )
        assert "SumFeature_V1V2" in result[0].to_pydict()
        for k, v in result[0].to_pydict().items():
            if k == "SumFeature_V1V2":
                assert v[0] == -2.378746582538124

    def test_aggregated_load_csv_with_path_given_to_feature(self) -> Any:
        f = Feature(
            name="sum_of_",
            options={"sum": ("V1", "V2")},
        )
        result = mloda.run_all(
            [f],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(files={self.file_path}),
        )
        assert "SumFeature_V1V2" in result[0].to_pydict()
        for k, v in result[0].to_pydict().items():
            if k == "SumFeature_V1V2":
                assert v[0] == -2.378746582538124

    def test_aggregated_load_csv_with_overwriting_match_feature_group_criteria(self) -> Any:
        _group = _overwritten_group()
        f = Feature(
            name="sum_of_",
            options={
                "sum": ("V1", "V2"),
                DOUBLING_MARKER: "dummy",
            },
        )
        result = mloda.run_all(
            [f],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(files={self.file_path}),
        )
        assert "SumFeature_V1V2" in result[0].to_pydict()
        for k, v in result[0].to_pydict().items():
            if k == "SumFeature_V1V2":
                assert v[0] == -4.757493165076248


class TestCsvFGOutsideTheRun:
    def test_calculate_feature_without_match_names_class_and_attribute(self) -> None:
        """An unmatched FeatureSet raises a ValueError naming the group class and input_data_match."""
        csv_group = load_group("csv_fg", "CsvFG")
        features = FeatureSet()
        features.add(Feature("unmatched_read_file_col"))

        with pytest.raises(ValueError, match=r"CsvFG") as excinfo:
            csv_group.calculate_feature(None, features)
        assert "input_data_match" in str(excinfo.value)

    def test_describe_columns_lists_the_header_without_types(self, tmp_path: Path) -> None:
        csv_group = load_group("csv_fg", "CsvFG")
        path = tmp_path / "described.csv"
        path.write_text("describe_id,describe_v1\n1,2\n")
        match = SourceMatch(source=os.path.abspath(path), access=str(path))

        assert csv_group.describe_columns(match) == {"describe_id": None, "describe_v1": None}

    def test_describe_columns_raises_not_implemented_when_the_file_cannot_enumerate(self, tmp_path: Path) -> None:
        csv_group = load_group("csv_fg", "CsvFG")
        path = tmp_path / "undescribable.csv"
        path.write_bytes(b"\xff\xfe\x00\x81" * 64)
        match = SourceMatch(source=os.path.abspath(path), access=str(path))

        with pytest.raises(NotImplementedError):
            csv_group.describe_columns(match)


class TestSameClassFGLinkWithDifferentDataSources:
    """Integration test: same FeatureGroup class linked with different data sources.

    This test verifies that left_discriminator/right_discriminator on Link correctly
    resolves two nodes of the same CsvFG subclass that load different CSV files.
    """

    file_path_a = f"{os.getcwd()}/tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"

    def test_left_discriminator_right_discriminator_resolves_same_class_fg_nodes(self) -> None:
        path_a = self.file_path_a
        csv_group = load_group("csv_fg", "CsvFG")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            path_b = f.name
            writer = csv.writer(f)
            writer.writerow(["id", "score"])
            for i, score in enumerate([100, 85, 92, 78, 95, 88, 91, 76, 83]):
                writer.writerow([i, score])

        try:

            class ReadFileWithIndex(csv_group):  # type: ignore[misc,valid-type]
                @classmethod
                def index_columns(cls) -> list[Index] | None:
                    return [Index(("id",))]

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: FeatureName | str,
                    options: Options,
                    data_access_collection: DataAccessCollection | None = None,
                ) -> bool:
                    if options.get("discriminator_test") is None:
                        return False
                    if isinstance(feature_name, FeatureName):
                        feature_name = str(feature_name)
                    return bool(super().match_feature_group_criteria(feature_name, options, data_access_collection))

            class JoinedCsvFeature(FeatureGroup):
                _path_a: str = path_a
                _path_b: str = path_b

                def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
                    _path_a = options.get("left_csv_path")
                    _path_b = options.get("right_csv_path")
                    link = Link.inner(
                        JoinSpec(ReadFileWithIndex, Index(("id",))),
                        JoinSpec(ReadFileWithIndex, Index(("id",))),
                        left_discriminator={"CsvFG": _path_a},
                        right_discriminator={"CsvFG": _path_b},
                    )
                    return {
                        Feature(
                            "id",
                            options={"CsvFG": _path_a, "discriminator_test": True},
                        ),
                        Feature(
                            "V1",
                            link=link,
                            index=Index(("id",)),
                            options={"CsvFG": _path_a, "discriminator_test": True},
                        ),
                        Feature(
                            "id",
                            options={"CsvFG": _path_b, "discriminator_test": True},
                        ),
                        Feature(
                            "score",
                            index=Index(("id",)),
                            options={"CsvFG": _path_b, "discriminator_test": True},
                        ),
                    }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    v1 = data.column("V1")
                    score = data.column("score").cast(pa.float64())
                    combined = pc.add(v1, score)
                    return pa.table({"JoinedCsvFeature": combined})

                @classmethod
                def feature_names_supported(cls) -> set[str]:
                    return {"JoinedCsvFeature"}

            result = mloda.run_all(
                [Feature("JoinedCsvFeature", options={"left_csv_path": path_a, "right_csv_path": path_b})],
                compute_frameworks=["PyArrowTable"],
                plugin_collector=PluginCollector.enabled_feature_groups({ReadFileWithIndex, JoinedCsvFeature}),
            )
            assert "JoinedCsvFeature" in result[0].to_pydict()
            assert len(result[0].to_pydict()["JoinedCsvFeature"]) > 0
        finally:
            os.remove(path_b)

    def test_missing_discriminator_raises_helpful_error(self) -> None:
        """Same-class FG link without discriminators raises a clear error message."""
        path_a = self.file_path_a
        csv_group = load_group("csv_fg", "CsvFG")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            path_b = f.name
            writer = csv.writer(f)
            writer.writerow(["id", "score"])
            for i, score in enumerate([100, 85, 92, 78, 95, 88, 91, 76, 83]):
                writer.writerow([i, score])

        try:

            class ReadFileWithIndexNoDisc(csv_group):  # type: ignore[misc,valid-type]
                @classmethod
                def index_columns(cls) -> list[Index] | None:
                    return [Index(("id",))]

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: FeatureName | str,
                    options: Options,
                    data_access_collection: DataAccessCollection | None = None,
                ) -> bool:
                    if options.get("no_disc_test") is None:
                        return False
                    if isinstance(feature_name, FeatureName):
                        feature_name = str(feature_name)
                    return bool(super().match_feature_group_criteria(feature_name, options, data_access_collection))

            class JoinedCsvNoDisc(FeatureGroup):
                _path_a: str = path_a
                _path_b: str = path_b

                def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
                    _path_a = options.get("left_csv_path")
                    _path_b = options.get("right_csv_path")
                    link = Link.inner(
                        JoinSpec(ReadFileWithIndexNoDisc, Index(("id",))),
                        JoinSpec(ReadFileWithIndexNoDisc, Index(("id",))),
                    )
                    return {
                        Feature(
                            "id",
                            options={"CsvFG": _path_a, "no_disc_test": True},
                        ),
                        Feature(
                            "V1",
                            link=link,
                            index=Index(("id",)),
                            options={"CsvFG": _path_a, "no_disc_test": True},
                        ),
                        Feature(
                            "id",
                            options={"CsvFG": _path_b, "no_disc_test": True},
                        ),
                        Feature(
                            "score",
                            index=Index(("id",)),
                            options={"CsvFG": _path_b, "no_disc_test": True},
                        ),
                    }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

                @classmethod
                def feature_names_supported(cls) -> set[str]:
                    return {"JoinedCsvNoDisc"}

            with pytest.raises((ValueError, Exception), match="left_discriminator"):
                mloda.run_all(
                    [Feature("JoinedCsvNoDisc", options={"left_csv_path": path_a, "right_csv_path": path_b})],
                    compute_frameworks=["PyArrowTable"],
                    plugin_collector=PluginCollector.enabled_feature_groups({ReadFileWithIndexNoDisc, JoinedCsvNoDisc}),
                )
        finally:
            os.remove(path_b)
