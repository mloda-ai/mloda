import importlib
import os
from typing import Any

import pytest

from mloda.core.abstract_plugins.components import declaration_surface
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda.provider import DataCreatorFeatureGroup
from mloda.user import mloda

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401
from tests.test_core.test_integration.test_core.test_runner_one_compute_framework import SumFeature  # noqa: F401


class InputDataTestFeatureGroup(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {
            "InputDataTestFeatureGroup_id",
            "InputDataTestFeatureGroup_V1",
            "InputDataTestFeatureGroup_V2",
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "InputDataTestFeatureGroup_id": [12, 2, 3],
            "InputDataTestFeatureGroup_V1": [1, 2, 3],
            "InputDataTestFeatureGroup_V2": [1, 2, 3],
        }


class TestInputData:
    file_path = f"{os.path.dirname(os.path.abspath(__file__))}/creditcard_2023.csv"

    feature_names = "id,V1,V2"
    feature_list = feature_names.split(",")

    def test_data_creator(self) -> Any:
        features = [f"InputDataTestFeatureGroup_{f}" for f in self.feature_list]
        result = mloda.run_all(features, compute_frameworks=["PyArrowTable"])
        assert result[0].to_pydict() == {
            "InputDataTestFeatureGroup_id": [12, 2, 3],
            "InputDataTestFeatureGroup_V1": [1, 2, 3],
            "InputDataTestFeatureGroup_V2": [1, 2, 3],
        }

    def test_api_data(self) -> None:
        features = [f"ApiDataTestFeatureGroup_{f}" for f in self.feature_list]

        result = mloda.run_all(
            features,
            compute_frameworks=["PyArrowTable"],
            api_data={
                "TestApiData": {
                    "ApiDataTestFeatureGroup_id": [12, 2, 3],
                    "ApiDataTestFeatureGroup_V1": [1, 2, 3],
                    "ApiDataTestFeatureGroup_V2": [1, 2, 3],
                }
            },
        )

        assert result[0].to_pydict() == {
            "ApiDataTestFeatureGroup_id": [12, 2, 3],
            "ApiDataTestFeatureGroup_V1": [1, 2, 3],
            "ApiDataTestFeatureGroup_V2": [1, 2, 3],
        }

    def test_agg_creator_and_api_data(self) -> None:
        _agg_creator = Feature(
            name="sum_of_",
            options={
                "sum": ("InputDataTestFeatureGroup_id", "InputDataTestFeatureGroup_V1"),
            },
        )
        _agg_api_input_feature = Feature(
            name="sum_of_",
            options={
                "sum": ("ApiDataTestFeatureGroup_V1", "ApiDataTestFeatureGroup_V2"),
            },
        )

        result = mloda.run_all(
            [_agg_creator, _agg_api_input_feature],
            compute_frameworks=["PyArrowTable"],
            api_data={
                "TestApiData": {
                    "ApiDataTestFeatureGroup_id": [12, 2, 3],
                    "ApiDataTestFeatureGroup_V1": [1, 2, 3],
                    "ApiDataTestFeatureGroup_V2": [1, 2, 3],
                }
            },
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
                    "SumFeature_ApiDataTestFeatureGroup_V1ApiDataTestFeatureGroup_V2": [12, 12, 12],
                }
            )


_REMOVED_MODULE_SYMBOLS = [
    ("mloda.core.abstract_plugins.components.input_data.reader_probe", "probe_reader"),
    ("mloda.core.abstract_plugins.components.match_rejection", "INPUT_DATA_OWNED_STAGE"),
    ("mloda.core.abstract_plugins.components.match_rejection", "restamp_match_rejections_since"),
    ("mloda.core.abstract_plugins.components.declared_attributes", "current_declaration_requirement"),
    ("mloda.core.abstract_plugins.components.declared_attributes", "declaration_requirement_scope"),
    ("mloda.core.abstract_plugins.components.credential_scrub", "_URI_PATTERN"),
    ("mloda.provider", "DataCreator"),
    ("mloda.provider", "BaseInputData"),
]


class TestInputDataPathIsGone:
    """BaseInputData and the FeatureGroup input_data matching path are removed."""

    @pytest.mark.parametrize(("module", "symbol"), _REMOVED_MODULE_SYMBOLS)
    def test_the_legacy_module_symbol_is_gone(self, module: str, symbol: str) -> None:
        assert not hasattr(importlib.import_module(module), symbol)

    def test_the_reader_declaration_surface_is_gone(self) -> None:
        assert not hasattr(declaration_surface, "DeclarationSurface")

    def test_the_base_input_data_module_is_gone(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("mloda.core.abstract_plugins.components.input_data.base_input_data")

    @pytest.mark.parametrize("member", ["input_data", "_matches_input_data", "_is_root_and_matches_input_data"])
    def test_feature_group_has_no_input_data_member(self, member: str) -> None:
        assert not hasattr(FeatureGroup, member)
