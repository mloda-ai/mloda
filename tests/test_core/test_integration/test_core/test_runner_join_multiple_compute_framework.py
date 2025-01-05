from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
import pytest
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_core.core.engine import Engine
from mloda_core.runtime.run import Runner
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import Link


class SecondCfw(PyarrowTable):
    pass


class ThirdCfw(PyarrowTable):
    pass


class FourthCfw(PyarrowTable):
    pass


class JoinCfwTest1(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3], "idx": ["a", "b", "c"]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyarrowTable}


class JoinCfwTest2(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [4, 5, 6], "idx": ["a", "b", "c"]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {SecondCfw}


class JoinCfwTest3(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [7, 8, 9], "idx": ["a", "b", "c"]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {ThirdCfw}


class JoinCfwTest4(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [12, 13, 14], "idx": ["a", "b", "c"]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {FourthCfw}


############################################


class Join2CfwTest(AbstractFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.int32_of("JoinCfwTest1"), Feature.int32_of("JoinCfwTest2")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data.column("JoinCfwTest1")
        data.column("JoinCfwTest2")
        return {cls.get_class_name(): [10, 2, 3]}


class Join3CfwTest(AbstractFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.int32_of("JoinCfwTest1"), Feature.int32_of("JoinCfwTest2"), Feature.int32_of("JoinCfwTest3")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data.column("JoinCfwTest1")
        data.column("JoinCfwTest2")
        data.column("JoinCfwTest3")
        return {cls.get_class_name(): [33, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyarrowTable}


class Join4CfwTest(Join3CfwTest):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature.int32_of("JoinCfwTest1"),
            Feature.int32_of("JoinCfwTest2"),
            Feature.int32_of("JoinCfwTest3"),
            Feature.int32_of("JoinCfwTest4"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data.column("JoinCfwTest1")
        data.column("JoinCfwTest2")
        data.column("JoinCfwTest3")
        data.column("JoinCfwTest4")
        return {cls.get_class_name(): [33, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyarrowTable}


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationModes.SYNC}),
        ({ParallelizationModes.THREADING}),
        # ({ParallelizationModes.MULTIPROCESSING}),
    ],
)
class TestEngineMultipleJoinCfw:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def basic_runner(
        self, features: Features, parallelization_modes: Set[ParallelizationModes], flight_server: Any, links: Set[Link]
    ) -> Runner:
        compute_framework: Set[Type[ComputeFrameWork]] = {
            PyarrowTable,
            SecondCfw,
            ThirdCfw,
            FourthCfw,
        }

        engine = Engine(features, compute_framework, links)

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

    def test_runner_join_multiple_cfw1_most_basic(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        def inner_loop(join_type: str) -> None:
            idx = Index(
                ("idx",),
            )

            left: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest1, idx)
            right: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest2, idx)

            links = {Link(join_type, left, right)}

            features = self.get_features(["Join2CfwTest"])
            runner = self.basic_runner(features, modes, flight_server, links)
            res = runner.get_result()
            assert res[0].to_pydict() == {"Join2CfwTest": [10, 2, 3]}

        for join_type in ["inner", "left", "right", "outer"]:
            inner_loop(join_type)

    def base_join_runner(
        self, modes: Set[ParallelizationModes], flight_server: Any, links: Set[Link], f_name: str = "3"
    ) -> None:
        f_name = f"Join{f_name}CfwTest"
        features = self.get_features([f_name])
        runner = self.basic_runner(features, modes, flight_server, links)
        res = runner.get_result()
        res = runner.get_result()
        assert res[0].to_pydict() == {f_name: [33, 2, 3]}

    def test_runner_join_multiple_cfw2_join_to_same_base(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest1, idx)
        right_1: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest2, idx)
        right_2: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest3, idx)
        links = {Link("inner", left, right_1), Link("inner", left, right_2)}

        self.base_join_runner(modes, flight_server, links)

    def test_runner_join_multiple_cfw3_join_to_right_base(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest1, idx)
        right_1: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest2, idx)
        right_2: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest3, idx)
        links = {Link("inner", right_1, left), Link("inner", right_2, left)}

        self.base_join_runner(modes, flight_server, links)

    def test_runner_join_multiple_cfw4_chained_join(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        idx = Index(
            ("idx",),
        )

        left: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest1, idx)
        right_1: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest2, idx)
        right_2: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest3, idx)
        links = {Link("inner", left, right_1), Link("inner", right_1, right_2)}

        self.base_join_runner(modes, flight_server, links)

    def test_runner_join_multiple_cfw5_double_chained_join(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest1, idx)
        right_1: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest2, idx)
        right_2: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest3, idx)
        right_3: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest4, idx)
        links = {Link("inner", left, right_1), Link("inner", right_1, right_2), Link("inner", right_2, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("inner", left, right_1), Link("inner", right_1, right_2), Link("inner", right_1, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

    def test_runner_join_multiple_cfw6_double_chained_join(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest1, idx)
        right_1: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest2, idx)
        right_2: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest3, idx)
        right_3: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest4, idx)
        links = {Link("inner", left, right_1), Link("inner", left, right_2), Link("inner", left, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("inner", left, right_1), Link("inner", right_2, left), Link("inner", left, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("inner", left, right_1), Link("inner", right_2, left), Link("inner", right_3, left)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

    def test_runner_join_multiple_cfw6_double_chained_join_left(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest1, idx)
        right_1: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest2, idx)
        right_2: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest3, idx)
        right_3: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest4, idx)

        links = {Link("left", left, right_1), Link("left", left, right_2), Link("left", left, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("left", left, right_1), Link("left", right_2, left), Link("left", left, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("outer", left, right_1), Link("inner", right_2, left), Link("inner", right_3, left)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

    def test_runner_join_multiple_cfw6_double_chained_join_right(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest1, idx)
        right_2: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest2, idx)
        right_3: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest3, idx)
        right_4: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest4, idx)

        links = {Link("left", left, right_2), Link("right", right_3, left), Link("left", left, right_4)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("inner", left, right_2), Link("right", left, right_3), Link("right", left, right_4)}
        with pytest.raises(ValueError):
            self.base_join_runner(modes, flight_server, links, f_name="4")

        with pytest.raises(ValueError):
            links = {Link("right", left, right_2), Link("inner", right_3, left), Link("right", right_4, left)}
            self.base_join_runner(modes, flight_server, links, f_name="4")

    def test_runner_join_multiple_cfw7_double_chained_join(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest1, idx)
        right_2: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest2, idx)
        right_3: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest3, idx)
        right_4: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest4, idx)

        links = {Link("inner", left, right_2), Link("inner", right_3, left), Link("inner", right_3, right_4)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("inner", right_2, left), Link("inner", right_4, right_3), Link("inner", right_2, right_4)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

    def test_raise_exception_if_duplicated_links_given(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        """This test should be moved to a unit test rather as it tests that Link.validate in engine init is working correctly. Was faster to test here."""

        idx = Index(
            ("idx",),
        )

        left: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest1, idx)
        right_2: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest2, idx)
        right_3: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest3, idx)
        right_4: Tuple[type[AbstractFeatureGroup], Index] = (JoinCfwTest4, idx)

        links = {
            Link("inner", right_2, left),
            Link("inner", right_4, right_3),
            Link("inner", right_3, right_4),
            Link("inner", right_2, right_4),
        }
        with pytest.raises(ValueError):
            self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {
            Link("inner", right_2, left),
            Link("right", right_3, right_4),
            Link("inner", right_3, right_4),
            Link("inner", right_2, right_4),
        }

        with pytest.raises(ValueError):
            self.base_join_runner(modes, flight_server, links, f_name="4")
