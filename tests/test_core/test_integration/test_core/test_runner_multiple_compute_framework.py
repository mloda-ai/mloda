from typing import Any, Dict, List, Optional, Set, Type, Union
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_core.core.engine import Engine
from mloda_core.runtime.run import Runner
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.core.step.feature_group_step import FeatureGroupStep
from mloda_core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options


class MultipleCfwTest1(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyArrowTable}


class SecondCfw(PyArrowTable):
    pass


class MultipleCfwTest2(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [4, 5, 6]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {SecondCfw}


class ChangeCfw(AbstractFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.int32_of("MultipleCfwTest1")}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {SecondCfw}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pc.multiply(data.column("MultipleCfwTest1"), 2)


class ThirdCfw(PyArrowTable):
    pass


class ChangeCfwThird(AbstractFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        # return {Feature.int32_of("ChangeCfw"), Feature.int32_of("MultipleCfwTest1")}
        return {Feature.int32_of("ChangeCfw")}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {ThirdCfw}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data.column("ChangeCfw")
        data.column("MultipleCfwTest1")
        return pc.multiply(data.column("ChangeCfw"), 2)


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationModes.SYNC}),
        ({ParallelizationModes.THREADING}),
        # ({ParallelizationModes.MULTIPROCESSING}),
    ],
)
class TestEngineMultipleCfw:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def basic_runner(
        self,
        features: Features,
        parallelization_modes: Set[ParallelizationModes],
        flight_server: Any,
    ) -> Runner:
        compute_framework: Set[Type[ComputeFrameWork]] = {PyArrowTable, SecondCfw, ThirdCfw}

        engine = Engine(features, compute_framework, None)

        if ParallelizationModes.MULTIPROCESSING in parallelization_modes:
            runner = engine.compute(flight_server)
        else:
            runner = engine.compute(None)

        assert runner is not None

        try:
            runner.__enter__(parallelization_modes)
            runner.compute()
            runner.__exit__(None, None, None)
        finally:
            try:
                runner.manager.shutdown()
            except Exception:  # nosec
                pass

        # make sure all datasets are dropped on server
        # flight_infos = FlightServer.list_flight_infos(flight_server.location)
        # assert len(flight_infos) == 0

        return runner

    def test_runner_two_cfws(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        features = self.get_features(["MultipleCfwTest1", "MultipleCfwTest2"])
        runner = self.basic_runner(features, modes, flight_server)

        used_cfws = {step.compute_framework for step in runner.execution_planner.execution_plan}  # type: ignore
        assert used_cfws == {PyArrowTable, SecondCfw}

        for result in runner.get_result():
            assert isinstance(result, pa.Table)
            res = result.to_pydict()
            if "MultipleCfwTest1" in res:
                assert res == {"MultipleCfwTest1": [1, 2, 3]}
            else:
                assert res == {"MultipleCfwTest2": [4, 5, 6]}

    def test_runner_change_cfw_basic(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        features = self.get_features(["ChangeCfw"])
        runner = self.basic_runner(features, modes, flight_server)

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
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        features = self.get_features(["ChangeCfw", "MultipleCfwTest1", "MultipleCfwTest2"])
        runner = self.basic_runner(features, modes, flight_server)

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
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        features = self.get_features(["ChangeCfwThird"])
        runner = self.basic_runner(features, modes, flight_server)

        res = runner.get_result()
        assert res[0].to_pydict() == {"ChangeCfwThird": [4, 8, 12]}
