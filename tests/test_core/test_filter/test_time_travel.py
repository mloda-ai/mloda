from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Type, Union
import pytest

import pyarrow as pa

from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.filter.global_filter import GlobalFilter
from mloda_core.runtime.flight.flight_server import FlightServer
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_core.core.engine import Engine
from mloda_core.runtime.run import Runner
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.feature_set import FeatureSet


class TimeTravelNegativeFilterTest(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.filters is not None:
            raise ValueError(f"Test Filter should not be found: {features.filters}.")

        return {cls.get_class_name(): [1, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyArrowTable}


class TimeTravelPositiveFilterTest(AbstractFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if len(features.filters) != 1:  # type: ignore
            raise ValueError("Test Filter not found")

        return pa.table(
            {
                cls.get_class_name(): [1, 2, 3],
                "reference_time": [
                    (datetime.now(tz=timezone.utc) - timedelta(days=10)).isoformat(),  # 10 days ago - within range
                    (datetime.now(tz=timezone.utc) - timedelta(days=30)).isoformat(),  # 30 days ago - outside range
                    (datetime.now(tz=timezone.utc) - timedelta(days=30)).isoformat(),  # 30 days ago - outside range
                ],
            }
        )

    @classmethod
    def match_feature_group_criteria(  # type: ignore
        cls,
        feature_name: Union[FeatureName, str],
        _: Options,
        _2=None,
    ) -> bool:
        if feature_name.name in ["TimeTravelPositiveFilterTest", "reference_time", "time_travel_filter"]:  # type: ignore
            return True
        return False


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationModes.SYNC}),
        ({ParallelizationModes.THREADING}),
        ({ParallelizationModes.MULTIPROCESSING}),
    ],
)
class TestTimeTravel:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def basic_runner(
        self,
        features: Features,
        parallelization_modes: Set[ParallelizationModes],
        flight_server: Any,
        global_filter: GlobalFilter,
    ) -> Runner:
        compute_framework: Set[Type[ComputeFrameWork]] = {PyArrowTable}

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

    def test_time_travel_neg(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        filter_test = "TimeTravelNegativeFilterTest"

        features = self.get_features([filter_test])

        global_filter = GlobalFilter()

        global_filter.add_time_and_time_travel_filters(
            event_from=datetime.now(tz=timezone.utc) - timedelta(days=20),
            event_to=datetime.now(tz=timezone.utc),
            valid_from=datetime.now(tz=timezone.utc) - timedelta(days=40),
            valid_to=datetime.now(tz=timezone.utc),
        )

        runner = self.basic_runner(features, modes, flight_server, global_filter)
        for result in runner.get_result():
            res = result.to_pydict()
            assert res == {filter_test: [1, 2, 3]}

    def test_time_travel_pos(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        filter_test = "TimeTravelPositiveFilterTest"

        features = self.get_features([filter_test])

        global_filter = GlobalFilter()

        global_filter.add_time_and_time_travel_filters(
            event_from=datetime.now(tz=timezone.utc) - timedelta(days=20),
            event_to=datetime.now(tz=timezone.utc),
        )
        runner = self.basic_runner(features, modes, flight_server, global_filter)
        for result in runner.get_result():
            res = result.to_pydict()
            assert res == {filter_test: [1]}
