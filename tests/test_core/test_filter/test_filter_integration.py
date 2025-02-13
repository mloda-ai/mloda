from typing import Any, Dict, List, Optional, Set, Type, Union
import pytest
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.filter.global_filter import GlobalFilter
from mloda_core.runtime.flight.flight_server import FlightServer
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_core.core.engine import Engine
from mloda_core.runtime.run import Runner
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork


class GlobalFilterBasicTest(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if len(features.filters) != 1:  # type: ignore
            raise ValueError("Test Filter not found")

        for filter in features.filters:  # type: ignore
            if (
                filter.__repr__()
                != """<SingleFilter(feature_name=GlobalFilterBasicTest, type=eq, parameters=(('value', 1),))>"""
            ):
                raise ValueError("Test Filter not found")
        return {cls.get_class_name(): [1, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyarrowTable}


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

        return {"GlobalFilterFromDifferentColumn1": [1, 2, 3]}


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

        return {"GlobalFilterHasDifferentNameTest": [1, 2, 3]}

    def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
        return FeatureName(name="GlobalFilterHasDifferentNameTest")


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationModes.SYNC}),
        ({ParallelizationModes.THREADING}),
        ({ParallelizationModes.MULTIPROCESSING}),
    ],
)
class TestGlobalFilter:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def basic_runner(
        self,
        features: Features,
        parallelization_modes: Set[ParallelizationModes],
        flight_server: Any,
        global_filter: GlobalFilter,
    ) -> Runner:
        compute_framework: Set[Type[ComputeFrameWork]] = {PyarrowTable}

        engine = Engine(features, compute_framework, None, global_filter=global_filter)

        if ParallelizationModes.MULTIPROCESSING in parallelization_modes:
            runner = engine.compute(flight_server)
        else:
            runner = engine.compute(None)

        assert runner is not None

        try:
            runner.__enter__(parallelization_modes, None)
            runner.compute()
            runner.__exit__(None, None, None)
        finally:
            try:
                runner.manager.shutdown()
            except Exception:  # nosec
                pass

        # make sure all datasets are dropped on server
        if ParallelizationModes.MULTIPROCESSING in parallelization_modes:
            flight_infos = FlightServer.list_flight_infos(flight_server.location)
            assert len(flight_infos) == 0

        return runner

    def test_basic_global_filter(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        global_filter_test_basic = "GlobalFilterBasicTest"

        features = self.get_features([global_filter_test_basic])

        global_filter = GlobalFilter()
        global_filter.add_filter(global_filter_test_basic, "eq", {"value": 1})

        runner = self.basic_runner(features, modes, flight_server, global_filter)

        for result in runner.get_result():
            res = result.to_pydict()
            assert res == {global_filter_test_basic: [1, 2, 3]}

        result_global_filter = runner.execution_planner.global_filter
        assert result_global_filter.filters == global_filter.filters  # type: ignore
        assert isinstance(result_global_filter, GlobalFilter)

        # test registration of used filters
        for key, value in result_global_filter.collection.items():
            assert issubclass(key[0], AbstractFeatureGroup)
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
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        base_feature_name = "GlobalFilterFromDifferentColumn"

        features = self.get_features([f"{base_feature_name}1"])

        global_filter = GlobalFilter()

        # We could have parametized this test, but we don t want to add too many tests for no reason.
        if ParallelizationModes.MULTIPROCESSING in modes:
            filter_feat: Union[str, Feature] = f"{base_feature_name}2"
        else:
            filter_feat = Feature(name=f"{base_feature_name}2")

        global_filter.add_filter(filter_feat, "eq", {"value": 1})
        self.basic_runner(features, modes, flight_server, global_filter)

    def test_global_filter_filter_has_different_name(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        base_feature_name = "GlobalFilterHasDifferentName"

        features = self.get_features([f"{base_feature_name}1"])

        global_filter = GlobalFilter()

        # We could have parametized this test, but we don t want to add too many tests for no reason.
        if ParallelizationModes.MULTIPROCESSING not in modes:
            filter_feat: Union[str, Feature] = f"{base_feature_name}2"
        else:
            filter_feat = Feature(name=f"{base_feature_name}2")

        global_filter.add_filter(filter_feat, "eq", {"value": 1})
        self.basic_runner(features, modes, flight_server, global_filter)
