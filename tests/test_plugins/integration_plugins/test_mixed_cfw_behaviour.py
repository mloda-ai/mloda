import os
from typing import Any, List, Set
import time

from mloda.user import PluginLoader
import pytest


from mloda.provider import FeatureGroup
from mloda.user import DataAccessCollection
from mloda.user import Feature
from mloda.user import Index
from mloda.user import Link, JoinSpec
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader

import logging

logger = logging.getLogger(__name__)


try:
    import pandas as pd
except ImportError:
    pd = None


class TestMixComputeFramework:
    file_path = f"{os.getcwd()}/tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"

    feature_names = "id,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class"
    feature_list = feature_names.split(",")

    PluginLoader().all()

    @classmethod
    def get_feature_list_from_local_scope(cls, features: List[str], path: str) -> List[str | Feature]:
        _feature_list: List[str | Feature] = []
        for feature in features:
            compute_framework = "PyArrowTable" if feature == "Amount" else "PandasDataFrame"
            _f = Feature(
                name=feature,
                options={CsvReader.__name__: path},
                compute_framework=compute_framework,
            )
            _feature_list.append(_f)
        return _feature_list

    def assert_file(self, result_data: Any, expected_cols: int = 30) -> None:
        assert isinstance(result_data, pd.DataFrame)
        assert len(result_data) == 9
        assert len(result_data.columns) == expected_cols

    def test_mix_cfw_can_work_in_parallel(self) -> None:
        feature_list = self.feature_list
        features = self.get_feature_list_from_local_scope(feature_list, self.file_path)
        result_data = mloda.run_all(features, compute_frameworks=["PandasDataFrame", "PyArrowTable"])

        for data in result_data:
            if isinstance(data, pd.DataFrame):
                self.assert_file(data)
            else:
                assert data.num_columns == 1

    @pytest.mark.order("first")
    @pytest.mark.parametrize(
        "modes",
        [
            ({ParallelizationMode.SYNC}),
            ({ParallelizationMode.THREADING}),
            # ({ParallelizationMode.MULTIPROCESSING}),
        ],
    )
    def test_mix_cfw_can_be_called_from_same_feature(
        self,
        modes: Set[ParallelizationMode],
        flight_server: Any,  # noqa: F811
    ) -> None:
        from tests.test_plugins.integration_plugins.features_for_testing import MixedCfwFeature

        if MixedCfwFeature is None:
            raise ValueError("MixedCfwFeature is not imported")

        feature = Feature(name="MixedCfwFeature", compute_framework="PandasDataFrame")

        idx = Index(
            ("id",),
        )

        left = JoinSpec(ReadFileFeature, idx)
        right = JoinSpec(ReadFileFeature, idx)
        links = {Link("inner", left, right)}

        result_data = mloda.run_all(
            [feature],
            links=links,
            data_access_collection=DataAccessCollection(files={self.file_path}),
            parallelization_modes=modes,
            flight_server=flight_server,
            plugin_collector=PluginCollector.enabled_feature_groups({MixedCfwFeature, ReadFileFeature}),
        )

        assert len(result_data[0]["MixedCfwFeature"]) == 9

        if modes == {ParallelizationMode.MULTIPROCESSING}:
            time.sleep(1)

    @pytest.mark.parametrize(
        "modes",
        [
            ({ParallelizationMode.SYNC}),
            ({ParallelizationMode.THREADING}),
            # ({ParallelizationMode.MULTIPROCESSING}),
        ],
    )
    def test_duplicate_feature_setup(
        self,
        modes: Set[ParallelizationMode],
        flight_server: Any,  # noqa: F811
    ) -> None:
        from tests.test_plugins.integration_plugins.features_for_testing import DuplicateFeatureSetup

        if DuplicateFeatureSetup is None:
            raise ValueError("DuplicateFeatureSetup is not imported")

        feature = Feature(name="DuplicateFeatureSetup", compute_framework="PandasDataFrame")

        idx = Index(
            ("id",),
        )

        left = JoinSpec(ReadFileFeature, idx)
        right = JoinSpec(ReadFileFeature, idx)
        links = {Link("inner", left, right, {"self_left_alias": "dummy"}, {"self_right_alias": "dummy"})}

        result_data = mloda.run_all(
            [feature],
            links=links,
            data_access_collection=DataAccessCollection(files={self.file_path}),
            parallelization_modes=modes,
            flight_server=flight_server,
            plugin_collector=PluginCollector.enabled_feature_groups({DuplicateFeatureSetup, ReadFileFeature}),
        )

        assert len(result_data[0]["DuplicateFeatureSetup"]) == 9
