import os
from typing import Any, List, Optional

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.api.base_api_data import BaseApiData
from mloda_core.abstract_plugins.components.input_data.api.api_input_data_collection import (
    ApiInputDataCollection,
)
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.api.request import mlodaAPI


class InputDataTestFeatureGroup(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(
            supports_features={
                "InputDataTestFeatureGroup_id",
                "InputDataTestFeatureGroup_V1",
                "InputDataTestFeatureGroup_V2",
            }
        )

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "InputDataTestFeatureGroup_id": [12, 2, 3],
            "InputDataTestFeatureGroup_V1": [1, 2, 3],
            "InputDataTestFeatureGroup_V2": [1, 2, 3],
        }


class DataTestApiInputData(BaseApiData):
    @classmethod
    def column_names(cls) -> List[str]:
        return [
            "ApiInputDataTestFeatureGroup_id",
            "ApiInputDataTestFeatureGroup_V1",
            "ApiInputDataTestFeatureGroup_V2",
        ]


class TestInputData:
    file_path = f"{os.path.dirname(os.path.abspath(__file__))}/creditcard_2023.csv"

    feature_names = "id,V1,V2"
    feature_list = feature_names.split(",")

    def test_data_creator(self) -> Any:
        features = [f"InputDataTestFeatureGroup_{f}" for f in self.feature_list]
        result = mlodaAPI.run_all(features, compute_frameworks=["PyarrowTable"])  # type: ignore
        assert result[0].to_pydict() == {
            "InputDataTestFeatureGroup_id": [12, 2, 3],
            "InputDataTestFeatureGroup_V1": [1, 2, 3],
            "InputDataTestFeatureGroup_V2": [1, 2, 3],
        }

    def test_api_input_data(self) -> None:
        features = [f"ApiInputDataTestFeatureGroup_{f}" for f in self.feature_list]

        api_input_data_collection = ApiInputDataCollection(registry={"TestApiInputData": DataTestApiInputData})

        api_data = {
            "TestApiInputData": {
                "ApiInputDataTestFeatureGroup_id": [12, 2, 3],
                "ApiInputDataTestFeatureGroup_V1": [1, 2, 3],
                "ApiInputDataTestFeatureGroup_V2": [1, 2, 3],
            }
        }

        result = mlodaAPI.run_all(
            features,  # type: ignore
            compute_frameworks=["PyarrowTable"],
            api_input_data_collection=api_input_data_collection,
            api_data=api_data,
        )

        assert result[0].to_pydict() == {
            "ApiInputDataTestFeatureGroup_id": [12, 2, 3],
            "ApiInputDataTestFeatureGroup_V1": [1, 2, 3],
            "ApiInputDataTestFeatureGroup_V2": [1, 2, 3],
        }

    def test_agg_creator_and_api_input_data(self) -> None:
        _agg_creator = Feature(
            name="sum_of_",
            options={
                "sum": ("InputDataTestFeatureGroup_id", "InputDataTestFeatureGroup_V1"),
            },
        )
        _agg_api_input_feature = Feature(
            name="sum_of_",
            options={
                "sum": ("ApiInputDataTestFeatureGroup_V1", "ApiInputDataTestFeatureGroup_V2"),
            },
        )

        api_data = {
            "TestApiInputData": {
                "ApiInputDataTestFeatureGroup_id": [12, 2, 3],
                "ApiInputDataTestFeatureGroup_V1": [1, 2, 3],
                "ApiInputDataTestFeatureGroup_V2": [1, 2, 3],
            }
        }

        result = mlodaAPI.run_all(
            [_agg_creator, _agg_api_input_feature],
            compute_frameworks=["PyarrowTable"],
            api_input_data_collection=ApiInputDataCollection(registry={"TestApiInputData": DataTestApiInputData}),
            api_data=api_data,
        )

        for res in result:
            assert (
                res.to_pydict()
                == {
                    "SumFeature_InputDataTestFeatureGroup_idInputDataTestFeatureGroup_V1": [23, 23, 23],
                }
            ) or (
                res.to_pydict()
                == {
                    "SumFeature_ApiInputDataTestFeatureGroup_V1ApiInputDataTestFeatureGroup_V2": [12, 12, 12],
                }
            )
