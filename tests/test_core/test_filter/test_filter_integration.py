from typing import Any, Dict, List, Optional, Set, Type, Union

import pyarrow as pa

from mloda.user import FeatureName
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.user import Options
from mloda.user import GlobalFilter
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda.user import ParallelizationMode
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import Features
from mloda.provider import FeatureSet
from mloda.provider import ComputeFramework
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_ALL


class GlobalFilterBasicTest(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if len(features.filters) != 1:  # type: ignore
            raise ValueError("Test Filter not found")

        for filter in features.filters:  # type: ignore
            if filter.filter_type != "equal" or filter.parameter.value != 1:
                raise ValueError("Test Filter not found")
        return pa.table({cls.get_class_name(): [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


class GlobalFilterFromDifferentColumnTest(GlobalFilterBasicTest):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"GlobalFilterFromDifferentColumn1", "GlobalFilterFromDifferentColumn2"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if len(features.filters) != 1:  # type: ignore
            raise ValueError("Test Filter not found")

        if "GlobalFilterFromDifferentColumn2" not in features.get_all_names():
            raise ValueError("Filter feature not found.")

        for feat in features.features:
            if feat.name.name == "GlobalFilterFromDifferentColumn2":
                if feat.initial_requested_data is not False:
                    raise ValueError("Filter should not lead to automatic requested data.")

        return pa.table({"GlobalFilterFromDifferentColumn1": [1, 2, 3], "GlobalFilterFromDifferentColumn2": [1, 2, 3]})


class GlobalFilterHasDifferentNameTest(GlobalFilterBasicTest):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"GlobalFilterHasDifferentName1", "GlobalFilterHasDifferentName2"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if len(features.filters) != 1:  # type: ignore
            raise ValueError("Test Filter not found")

        if next(iter(features.filters)).filter_feature.name != "GlobalFilterHasDifferentNameTest":  # type: ignore
            raise ValueError("Filter feature name is not equal to eature name.")

        if len(features.get_all_names()) != 1:
            raise ValueError("Filter feature is same like normal feature.")

        return pa.table({"GlobalFilterHasDifferentNameTest": [1, 2, 3], "GlobalFilterHasDifferentName2": [1, 2, 3]})

    def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
        return FeatureName(name="GlobalFilterHasDifferentNameTest")


@PARALLELIZATION_MODES_ALL
class TestGlobalFilter:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def test_basic_global_filter(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        global_filter_test_basic = "GlobalFilterBasicTest"

        features = self.get_features([global_filter_test_basic])

        global_filter = GlobalFilter()
        global_filter.add_filter(global_filter_test_basic, "equal", {"value": 1})

        runner = MlodaTestRunner.run_engine(
            features, parallelization_modes=modes, flight_server=flight_server, global_filter=global_filter
        )

        for result in runner.get_result():
            res = result.to_pydict()
            assert res == {global_filter_test_basic: [1]}

        result_global_filter = runner.execution_planner.global_filter
        assert result_global_filter.filters == global_filter.filters  # type: ignore
        assert isinstance(result_global_filter, GlobalFilter)

        # test registration of used filters
        for key, value in result_global_filter.collection.items():
            assert issubclass(key[0], FeatureGroup)
            assert isinstance(key[1], FeatureName)
            assert key[0].get_class_name() == global_filter_test_basic
            assert key[1].name == global_filter_test_basic

            assert isinstance(value, set)
            assert len(value) == 1

            single_feature = next(iter(value))
            assert single_feature.filter_feature.name == global_filter_test_basic
            assert single_feature.filter_feature.name == global_filter_test_basic
            assert next(iter(global_filter.filters)).uuid == single_feature.uuid

    def test_global_filter_filter_requests_other_column(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        base_feature_name = "GlobalFilterFromDifferentColumn"

        features = self.get_features([f"{base_feature_name}1"])

        global_filter = GlobalFilter()

        # We could have parametized this test, but we don t want to add too many tests for no reason.
        if ParallelizationMode.MULTIPROCESSING in modes:
            filter_feat: Union[str, Feature] = f"{base_feature_name}2"
        else:
            filter_feat = Feature(name=f"{base_feature_name}2")

        global_filter.add_filter(filter_feat, "equal", {"value": 1})
        runner = MlodaTestRunner.run_engine(
            features, parallelization_modes=modes, flight_server=flight_server, global_filter=global_filter
        )

        for result in runner.get_result():
            res = result.to_pydict()
            assert res == {"GlobalFilterFromDifferentColumn1": [1]}

    def test_global_filter_filter_has_different_name(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        base_feature_name = "GlobalFilterHasDifferentName"

        features = self.get_features([f"{base_feature_name}1"])

        global_filter = GlobalFilter()

        # We could have parametized this test, but we don t want to add too many tests for no reason.
        if ParallelizationMode.MULTIPROCESSING not in modes:
            filter_feat: Union[str, Feature] = f"{base_feature_name}2"
        else:
            filter_feat = Feature(name=f"{base_feature_name}2")

        global_filter.add_filter(filter_feat, "equal", {"value": 1})
        runner = MlodaTestRunner.run_engine(
            features, parallelization_modes=modes, flight_server=flight_server, global_filter=global_filter
        )

        for result in runner.get_result():
            res = result.to_pydict()
            assert res == {"GlobalFilterHasDifferentNameTest": [1]}
