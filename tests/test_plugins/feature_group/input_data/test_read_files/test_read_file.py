import os
from typing import Any, Dict, List, Optional, Tuple, Union

from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
import pyarrow as pa

from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.api.request import mlodaAPI


class OverwrittenReadCsvInputDataTestFeatureGroup(ReadFileFeature):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        feature_names = "id,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class"
        feature_list = feature_names.split(",")

        # We added this, because else this feature name rule would supersede the other examples.
        if options.data.get("OverwrittenReadCsvInputDataTestFeatureGroup", None) is None:
            return False

        if feature_name in feature_list:
            return True

        return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        reader = cls.input_data()
        if reader is not None:
            result = reader.load(features)

            new_columns = {
                col_name: pa.array([value * 2 for value in result[col_name].to_pylist()])
                for col_name in result.schema.names
            }
            return pa.table(new_columns)
        raise ValueError(f"Reading file failed for feature {features.get_name_of_one_feature()}.")


class TestInputData:
    file_path = f"{os.getcwd()}/tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"

    feature_names = "id,V1,V2"
    feature_list = feature_names.split(",")

    @classmethod
    def get_features(
        cls, features: List[str], path: Optional[str] = None, additional_options: Dict[str, Any] = {}
    ) -> List[str | Feature]:
        _feature_list: List[str | Feature] = []
        for feature in features:
            _f = Feature(name=feature)
            for k, v in additional_options.items():
                _f.options.add(k, v)
            if path is not None:
                _f.options.add(CsvReader.__name__, path)
            _feature_list.append(_f)
        return _feature_list

    def test_local_scope_file(self) -> Any:
        features = self.get_features(self.feature_list, self.file_path)
        result = mlodaAPI.run_all(features, compute_frameworks=["PyarrowTable"])
        assert "V2" in result[0].to_pydict()

    def test_local_scope_folder(self) -> Any:
        file_path = self.file_path.replace("creditcard_2023_short.csv", "")
        features = self.get_features(self.feature_list, file_path)
        result = mlodaAPI.run_all(features, compute_frameworks=["PyarrowTable"])
        assert "V2" in result[0].to_pydict()

    def test_global_scope_file(self) -> Any:
        result = mlodaAPI.run_all(
            self.feature_list,  # type: ignore
            compute_frameworks=["PyarrowTable"],
            data_access_collection=DataAccessCollection(files={self.file_path}),
        )
        assert "V2" in result[0].to_pydict()

    def test_global_scope_folder(self) -> Any:
        file_path = self.file_path.replace("creditcard_2023_short.csv", "")
        result = mlodaAPI.run_all(
            self.feature_list,  # type: ignore
            compute_frameworks=["PyarrowTable"],
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
        This test checks if the overwritten match feature group criteria is applied if overriden.

        Further, this checks if sub_classes are filtered out correctly. (Functionality in IdentifyFeatureGroupClass).
        """
        features = self.get_features(self.feature_list, None, {"OverwrittenReadCsvInputDataTestFeatureGroup": "dummy"})

        result = mlodaAPI.run_all(
            features,
            compute_frameworks=["PyarrowTable"],
            data_access_collection=DataAccessCollection(files={self.file_path}),
        )
        assert "V2" in result[0].to_pydict()
        for k, v in result[0].to_pydict().items():
            if k == "id":
                assert v == [0, 2, 4, 6, 8, 10, 12, 14, 16]

    def test_overwriting_match_feature_group_criteria_using_local_scope(self) -> Any:
        """
        This test checks if the overwritten match feature group criteria is applied if overriden.

        Further, this checks if sub_classes are filtered out correctly. (Functionality in IdentifyFeatureGroupClass).
        """
        features = self.get_features(
            self.feature_list, self.file_path, {"OverwrittenReadCsvInputDataTestFeatureGroup": "dummy"}
        )

        result = mlodaAPI.run_all(
            features,
            compute_frameworks=["PyarrowTable"],
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
        result = mlodaAPI.run_all(
            [f],
            compute_frameworks=["PyarrowTable"],
            data_access_collection=DataAccessCollection(folders={file_path}),
        )
        assert "SumFeature_V1V2" in result[0].to_pydict()
        for k, v in result[0].to_pydict().items():
            if k == "SumFeature_V1V2":
                assert v[0] == -2.378746582538124

    def test_aggregated_load_csv_with_path_given_to_feature(self) -> Any:
        f = Feature(
            name="sum_of_",
            options={"sum": ("V1", "V2"), CsvReader.__name__: self.file_path},
        )
        result = mlodaAPI.run_all(
            [f],
            compute_frameworks=["PyarrowTable"],
        )
        assert "SumFeature_V1V2" in result[0].to_pydict()
        for k, v in result[0].to_pydict().items():
            if k == "SumFeature_V1V2":
                assert v[0] == -2.378746582538124

    def test_aggregated_load_csv_with_overwriting_match_feature_group_criteria(self) -> Any:
        f = Feature(
            name="sum_of_",
            options={
                "sum": ("V1", "V2"),
                CsvReader.__name__: self.file_path,
                "OverwrittenReadCsvInputDataTestFeatureGroup": "dummy",
            },
        )
        result = mlodaAPI.run_all(
            [f],
            compute_frameworks=["PyarrowTable"],
        )
        assert "SumFeature_V1V2" in result[0].to_pydict()
        for k, v in result[0].to_pydict().items():
            if k == "SumFeature_V1V2":
                assert v[0] == -4.757493165076248


class TestReadFile:
    def test_validate_columns(self) -> None:
        class TestReadFile(ReadFile):
            @classmethod
            def get_column_names(cls, file_name: str) -> List[str]:
                return ["id", "V1", "V2"]

            @classmethod
            def suffix(cls) -> Tuple[str, ...]:
                return (".csv",)

        assert TestReadFile.validate_columns("dummy.csv", ["id", "V1"])
        assert not TestReadFile.validate_columns("dummy.csv", ["id", "V3"])

    def test_match_read_file_data_access(self) -> None:
        class TestReadFile(ReadFile):
            @classmethod
            def get_column_names(cls, file_name: str) -> List[str]:
                return ["id", "V1", "V2"]

            @classmethod
            def suffix(cls) -> Tuple[str, ...]:
                return (".csv",)

        data_accesses = ["dummy.csv", "dummy2.csv"]
        feature_names = ["id", "V1"]
        assert TestReadFile.match_read_file_data_access(data_accesses, feature_names) == "dummy.csv"

    def test_match_subclass_data_access(self) -> None:
        class TestReadFile(ReadFile):
            @classmethod
            def get_column_names(cls, file_name: str) -> List[str]:
                return ["id", "V1", "V2"]

            @classmethod
            def suffix(cls) -> Tuple[str, ...]:
                return (".csv",)

        data_access = DataAccessCollection(files={"dummy.csv"})
        feature_names = ["id", "V1"]
        assert TestReadFile.match_subclass_data_access(data_access, feature_names) == "dummy.csv"

    def test_init_reader(self) -> None:
        class TestReadFile(ReadFile):
            @classmethod
            def get_column_names(cls, file_name: str) -> List[str]:
                return ["id", "V1", "V2"]

            @classmethod
            def suffix(cls) -> Tuple[str, ...]:
                return (".csv",)

        options = Options(data={"BaseInputData": (TestReadFile, "dummy.csv")})
        reader, data_access = TestReadFile().init_reader(options)
        assert isinstance(reader, TestReadFile)
        assert data_access == "dummy.csv"

    def test_load(self) -> None:
        class TestReadFile(ReadFile):
            @classmethod
            def get_column_names(cls, file_name: str) -> List[str]:
                return ["id", "V1", "V2"]

            @classmethod
            def suffix(cls) -> Tuple[str, ...]:
                return (".csv",)

            @classmethod
            def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
                return pa.table({"id": [1, 2], "V1": [3, 4], "V2": [5, 6]})

        options = Options(data={"BaseInputData": (TestReadFile, "dummy.csv")})
        features = FeatureSet()
        features.add(Feature("id", options=options))
        features.add(Feature("V1", options=options))
        features.add(Feature("V2", options=options))
        data = TestReadFile().load(features)
        assert data.column_names == ["id", "V1", "V2"]
