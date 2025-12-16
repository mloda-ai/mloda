from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Type, Union

import pyarrow as pa

from mloda.user import FeatureName
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda import Options
from mloda import ComputeFramework
from mloda.user import GlobalFilter
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda.user import ParallelizationMode
from mloda import FeatureGroup
from mloda import Feature
from mloda.user import Features
from mloda.provider import FeatureSet
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_ALL


class TimeTravelNegativeFilterTest(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.filters is not None:
            raise ValueError(f"Test Filter should not be found: {features.filters}.")

        return {cls.get_class_name(): [1, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


class TimeTravelPositiveFilterTest(FeatureGroup):
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


@PARALLELIZATION_MODES_ALL
class TestTimeTravel:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def test_time_travel_neg(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        filter_test = "TimeTravelNegativeFilterTest"

        features = self.get_features([filter_test])

        global_filter = GlobalFilter()

        global_filter.add_time_and_time_travel_filters(
            event_from=datetime.now(tz=timezone.utc) - timedelta(days=20),
            event_to=datetime.now(tz=timezone.utc),
            valid_from=datetime.now(tz=timezone.utc) - timedelta(days=40),
            valid_to=datetime.now(tz=timezone.utc),
        )

        runner = MlodaTestRunner.run_engine(
            features, parallelization_modes=modes, flight_server=flight_server, global_filter=global_filter
        )
        for result in runner.get_result():
            res = result.to_pydict()
            assert res == {filter_test: [1, 2, 3]}

    def test_time_travel_pos(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        filter_test = "TimeTravelPositiveFilterTest"

        features = self.get_features([filter_test])

        global_filter = GlobalFilter()

        global_filter.add_time_and_time_travel_filters(
            event_from=datetime.now(tz=timezone.utc) - timedelta(days=20),
            event_to=datetime.now(tz=timezone.utc),
        )
        runner = MlodaTestRunner.run_engine(
            features, parallelization_modes=modes, flight_server=flight_server, global_filter=global_filter
        )
        for result in runner.get_result():
            res = result.to_pydict()
            assert res == {filter_test: [1]}
