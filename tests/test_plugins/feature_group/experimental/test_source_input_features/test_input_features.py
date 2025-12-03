from pathlib import Path
from typing import Any, List, Optional, Set

import pandas as pd

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.data_access_collection import (
    DataAccessCollection,
)
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.input_data.base_input_data import (
    BaseInputData,
)
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import (
    DataCreator,
)
from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import (
    PlugInCollector,
)
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import (
    PandasDataFrame,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.source_input_feature import SourceInputFeature

from mloda_plugins.feature_group.input_data.api_data.api_data import ApiInputDataFeature
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader


class FeatureInputFeatureTest(AbstractFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["TestValue", "TestValue2"]}

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {cls.get_class_name()}


class InputFeatureGroupTest(SourceInputFeature):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["InputFeatureGroupTest"] = len(data)
        return data


class InputFeatureMergeTest(SourceInputFeature):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["InputFeatureMergeTest"] = data["FeatureInputCsv"].fillna("") + data["FeatureInputFeatureTest"].fillna("")
        return data


class TestInputFeatures:
    class FeatureInputCreatorTest(AbstractFeatureGroup):
        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return DataCreator({cls.get_class_name()})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {cls.get_class_name(): ["TestValue5", "TestValue6"]}

    _requested_name = "InputFeatureGroupTest"
    _enabled = PlugInCollector.enabled_feature_groups(
        {
            ApiInputDataFeature,
            InputFeatureGroupTest,
            FeatureInputFeatureTest,
            FeatureInputCreatorTest,
            ReadFileFeature,
        }
    )

    def test_input_feature_feature(self) -> None:
        feature_list: List[str | Feature] = []

        feature_list.append(
            Feature(
                name=self._requested_name,
                options={
                    DefaultOptionKeys.in_features: frozenset(["FeatureInputFeatureTest"]),
                    "initial_requested_data": True,
                },
            )
        )

        result = mlodaAPI.run_all(
            feature_list,
            plugin_collector=self._enabled,
            compute_frameworks={PandasDataFrame},
        )

        for res in result:
            assert len(res) == 2
            if "InputFeatureGroupTest" in res:
                assert list(res["InputFeatureGroupTest"].values) == [2, 2]
            else:
                assert set(res["FeatureInputFeatureTest"].values) == set(["TestValue", "TestValue2"])
        assert len(result) == 2

    def test_input_feature_api(self) -> None:
        """
        This test is a bit more complex, as it requires the use of the API.

        We pass api_data directly to run_all() - the ApiInputDataCollection is created internally.
        """
        feature_list: List[Feature | str] = []
        feature_list.append(
            Feature(
                name=self._requested_name,
                options={
                    DefaultOptionKeys.in_features: frozenset(["FeatureInputAPITest"]),
                    "initial_requested_data": True,
                },
            )
        )

        result = mlodaAPI.run_all(
            feature_list,
            plugin_collector=self._enabled,
            compute_frameworks={PandasDataFrame},
            api_data={"Example": {"FeatureInputAPITest": ["TestValue3", "TestValue4"]}},
        )
        for res in result:
            assert len(res) == 2
            if "InputFeatureGroupTest" in res:
                assert list(res["InputFeatureGroupTest"].values) == [2, 2]
            else:
                assert set(res["FeatureInputAPITest"].values) == set(["TestValue3", "TestValue4"])
        assert len(result) == 2

    def test_input_feature_data_creator(self) -> None:
        feature_list: List[Feature | str] = []
        feature_list.append(
            Feature(
                name=self._requested_name,
                options={
                    DefaultOptionKeys.in_features: frozenset(["FeatureInputCreatorTest"]),
                    "initial_requested_data": True,
                },
            )
        )

        result = mlodaAPI.run_all(
            feature_list,
            plugin_collector=self._enabled,
            compute_frameworks={PandasDataFrame},
        )
        for res in result:
            assert len(res) == 2
            if "InputFeatureGroupTest" in res:
                assert list(res["InputFeatureGroupTest"].values) == [2, 2]
            else:
                assert set(res["FeatureInputCreatorTest"].values) == set(["TestValue5", "TestValue6"])
        assert len(result) == 2

    def test_input_feature_feature_scope(self) -> None:
        file_path = Path(__file__).resolve().parent

        feature_list: List[Feature | str] = []
        feature_list.append(
            Feature(
                name=self._requested_name,
                options={
                    DefaultOptionKeys.in_features: frozenset([("FeatureInputCsv", CsvReader, file_path)]),
                    "initial_requested_data": True,
                },
            )
        )

        result = mlodaAPI.run_all(feature_list, compute_frameworks=["PandasDataFrame"])
        for res in result:
            assert len(res) == 2
            if "InputFeatureGroupTest" in res:
                assert list(res["InputFeatureGroupTest"].values) == [2, 2]
            else:
                assert set(res["FeatureInputCsv"].values) == set(["value9", "value7"])
        assert len(result) == 2

    def test_input_feature_global_scope(self) -> None:
        file_path = Path(__file__).resolve().parent
        file_path = file_path.joinpath("example.csv")
        data_access_collection = DataAccessCollection(files={str(file_path)})

        feature_list: List[Feature | str] = []
        feature_list.append(
            Feature(
                name=self._requested_name,
                options={
                    DefaultOptionKeys.in_features: frozenset(["FeatureInputCsv"]),
                    "initial_requested_data": True,
                },
            )
        )

        result = mlodaAPI.run_all(
            feature_list, compute_frameworks=["PandasDataFrame"], data_access_collection=data_access_collection
        )
        for res in result:
            assert len(res) == 2
            if "InputFeatureGroupTest" in res:
                assert list(res["InputFeatureGroupTest"].values) == [2, 2]
            else:
                assert set(res["FeatureInputCsv"].values) == set(["value9", "value7"])
        assert len(result) == 2

    def test_input_feature_all(self) -> None:
        file_path = Path(__file__).resolve().parent
        other_path = file_path.joinpath("example.csv")
        data_access_collection = DataAccessCollection(files={str(other_path)})
        # global scope
        feature_list: List[Feature | str] = []
        feature_list.append(
            Feature(
                name=self._requested_name,
                options={
                    DefaultOptionKeys.in_features: frozenset(["FeatureInputCsv"]),
                    "initial_requested_data": True,
                },
            )
        )

        # local scope
        feature_list.append(
            Feature(
                name=self._requested_name,
                options={
                    DefaultOptionKeys.in_features: frozenset([("FeatureInputCsv2", CsvReader, file_path)]),
                    "initial_requested_data": True,
                },
            )
        )

        # creator
        feature_list.append(
            Feature(
                name=self._requested_name,
                options={
                    DefaultOptionKeys.in_features: frozenset(["FeatureInputCreatorTest"]),
                    "initial_requested_data": True,
                },
            )
        )

        # api
        feature_list.append(
            Feature(
                name=self._requested_name,
                options={
                    DefaultOptionKeys.in_features: frozenset(["FeatureInputAPITest"]),
                    "initial_requested_data": True,
                },
            )
        )

        # feature
        feature_list.append(
            Feature(
                name=self._requested_name,
                options={
                    DefaultOptionKeys.in_features: frozenset(["FeatureInputFeatureTest"]),
                    "initial_requested_data": True,
                },
            )
        )

        result = mlodaAPI.run_all(
            feature_list,
            compute_frameworks=["PandasDataFrame"],
            data_access_collection=data_access_collection,
            api_data={"Example": {"FeatureInputAPITest": ["TestValue3", "TestValue4"]}},
        )
        assert len(result) == 10

    def test_input_feature_merge(self) -> None:
        requested_feature = "InputFeatureMergeTest"

        file_path = Path(__file__).resolve().parent
        file_path = file_path.joinpath("example.csv")
        data_access_collection = DataAccessCollection(files={str(file_path)})

        # global scope, feature
        feature_list: List[Feature | str] = []
        feature_list.append(
            Feature(
                name=requested_feature,
                options={
                    DefaultOptionKeys.in_features: frozenset(
                        [
                            (
                                "FeatureInputCsv",
                                None,
                                None,
                                (ReadFileFeature, "FeatureInputCsv"),
                                (FeatureInputFeatureTest, "FeatureInputFeatureTest"),
                                JoinType.APPEND.value,
                                "FeatureInputCsv",
                            ),
                            (
                                "FeatureInputFeatureTest",
                                None,
                                None,
                                None,
                                None,
                                None,
                                Index(("FeatureInputFeatureTest",)),
                            ),
                        ]
                    ),
                },
            )
        )

        result = mlodaAPI.run_all(
            feature_list,
            compute_frameworks=["PandasDataFrame"],
            data_access_collection=data_access_collection,
        )

        assert len(result) == 1
        assert all(
            result[0]
            == pd.DataFrame(
                {
                    "InputFeatureMergeTest": ["value7", "value9", "TestValue", "TestValue2"],
                }
            )
        )

    def test_input_feature_outer(self) -> None:
        requested_feature = "InputFeatureMergeTest"

        file_path = Path(__file__).resolve().parent
        file_path = file_path.joinpath("example.csv")
        data_access_collection = DataAccessCollection(files={str(file_path)})

        # global scope, feature
        feature_list: List[Feature | str] = []
        feature_list.append(
            Feature(
                name=requested_feature,
                options={
                    DefaultOptionKeys.in_features: frozenset(
                        [
                            (
                                "FeatureInputCsv",
                                None,
                                None,
                                (ReadFileFeature, "FeatureInputCsv"),
                                (FeatureInputFeatureTest, "FeatureInputFeatureTest"),
                                "outer",
                                "FeatureInputCsv",
                            ),
                            (
                                "FeatureInputFeatureTest",
                                None,
                                None,
                                None,
                                None,
                                None,
                                Index(("FeatureInputFeatureTest",)),
                            ),
                        ]
                    ),
                },
            )
        )

        result = mlodaAPI.run_all(
            feature_list,
            compute_frameworks=["PandasDataFrame"],
            data_access_collection=data_access_collection,
        )

        assert len(result) == 1
        assert all(
            result[0]
            == pd.DataFrame(
                {
                    "InputFeatureMergeTest": ["value7", "value9", "TestValue", "TestValue2"],
                }
            )
        )
