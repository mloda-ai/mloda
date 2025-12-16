from typing import Any, Dict, List, Optional, Set, Type, Union
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda import FeatureGroup
from mloda import Feature
from mloda.user import Features
from mloda.provider import FeatureSet
from mloda import Options
from tests.test_plugins.compute_framework.test_tooling.shared_compute_frameworks import (
    SecondCfw,
    ThirdCfw,
)
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_SYNC_THREADING


class MultipleCfwTest1(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


class MultipleCfwTest2(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [4, 5, 6]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SecondCfw}


class ChangeCfw(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.int32_of("MultipleCfwTest1")}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SecondCfw}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pc.multiply(data.column("MultipleCfwTest1"), 2)


class ChangeCfwThird(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        # return {Feature.int32_of("ChangeCfw"), Feature.int32_of("MultipleCfwTest1")}
        return {Feature.int32_of("ChangeCfw")}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {ThirdCfw}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data.column("ChangeCfw")
        data.column("MultipleCfwTest1")
        return pc.multiply(data.column("ChangeCfw"), 2)


COMPUTE_FRAMEWORKS: Set[Type[ComputeFramework]] = {PyArrowTable, SecondCfw, ThirdCfw}


@PARALLELIZATION_MODES_SYNC_THREADING
class TestEngineMultipleCfw:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def test_runner_two_cfws(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        features = self.get_features(["MultipleCfwTest1", "MultipleCfwTest2"])
        runner = MlodaTestRunner.run_engine(
            features, compute_frameworks=COMPUTE_FRAMEWORKS, parallelization_modes=modes, flight_server=flight_server
        )

        used_cfws = {step.compute_framework for step in runner.execution_planner.execution_plan}  # type: ignore
        assert used_cfws == {PyArrowTable, SecondCfw}

        for result in runner.get_result():
            assert isinstance(result, pa.Table)
            res = result.to_pydict()
            if "MultipleCfwTest1" in res:
                assert res == {"MultipleCfwTest1": [1, 2, 3]}
            else:
                assert res == {"MultipleCfwTest2": [4, 5, 6]}

    def test_runner_change_cfw_basic(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        features = self.get_features(["ChangeCfw"])
        runner = MlodaTestRunner.run_engine(
            features, compute_frameworks=COMPUTE_FRAMEWORKS, parallelization_modes=modes, flight_server=flight_server
        )

        execution_plan = runner.execution_planner.execution_plan
        used_cfws = [step.compute_framework for step in execution_plan if isinstance(step, FeatureGroupStep)]
        assert used_cfws == [PyArrowTable, SecondCfw]

        assert isinstance(execution_plan[1], TransformFrameworkStep)
        tfs = execution_plan[1]
        assert tfs.from_framework == PyArrowTable
        assert tfs.to_framework == SecondCfw
        assert tfs.from_feature_group == MultipleCfwTest1
        assert tfs.to_feature_group == ChangeCfw
        assert tfs.required_uuids == {execution_plan[0].features.any_uuid}  # type: ignore

        assert execution_plan[0].compute_framework == PyArrowTable  # type: ignore
        assert execution_plan[2].compute_framework == SecondCfw  # type: ignore

        result = runner.get_result()
        assert result[0].to_pydict() == {"ChangeCfw": [2, 4, 6]}

    def test_runner_change_cfw_first_framework_result(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        features = self.get_features(["ChangeCfw", "MultipleCfwTest1", "MultipleCfwTest2"])
        runner = MlodaTestRunner.run_engine(
            features, compute_frameworks=COMPUTE_FRAMEWORKS, parallelization_modes=modes, flight_server=flight_server
        )

        execution_plan = runner.execution_planner.execution_plan
        used_cfws = {step.compute_framework for step in execution_plan if isinstance(step, FeatureGroupStep)}
        assert used_cfws == {PyArrowTable, SecondCfw}

        for step in execution_plan:
            if isinstance(step, TransformFrameworkStep):
                tfs = step
                assert tfs.from_framework == PyArrowTable
                assert tfs.to_framework == SecondCfw
                assert tfs.from_feature_group == MultipleCfwTest1
                assert tfs.to_feature_group == ChangeCfw
            else:
                assert step.compute_framework in used_cfws  # type: ignore

        result = runner.get_result()
        for res in result:
            res = res.to_pydict()
            if "MultipleCfwTest1" in res:
                assert res == {"MultipleCfwTest1": [1, 2, 3]}
            elif "MultipleCfwTest2" in res:
                assert res == {"MultipleCfwTest2": [4, 5, 6]}
            else:
                assert res == {"ChangeCfw": [2, 4, 6]}

    def test_runner_change_cfw_third_framework_result(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        features = self.get_features(["ChangeCfwThird"])
        runner = MlodaTestRunner.run_engine(
            features, compute_frameworks=COMPUTE_FRAMEWORKS, parallelization_modes=modes, flight_server=flight_server
        )

        res = runner.get_result()
        assert res[0].to_pydict() == {"ChangeCfwThird": [4, 8, 12]}
