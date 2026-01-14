from typing import Any, Dict, List, Optional, Set, Type, Union
import pytest
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import Features
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda.user import Index
from mloda.user import Link, JoinSpec

# Import transformers to ensure they're registered
import mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_pyarrow_transformer  # noqa: F401
import mloda_plugins.compute_framework.base_implementations.pandas.pandaspyarrowtransformer  # noqa: F401
import mloda_plugins.compute_framework.base_implementations.polars.polars_pyarrow_transformer  # noqa: F401
import mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_pyarrow_transformer  # noqa: F401

from tests.test_plugins.compute_framework.test_tooling.shared_compute_frameworks import (
    SecondCfw,
    ThirdCfw,
    FourthCfw,
)
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_SYNC_THREADING


COMPUTE_FRAMEWORKS: Set[Type[ComputeFramework]] = {
    PyArrowTable,
    SecondCfw,
    ThirdCfw,
    FourthCfw,
    PandasDataFrame,
    PolarsLazyDataFrame,
    PythonDictFramework,
}


class JoinCfwTest1(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3], "idx": ["a", "b", "c"]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


class JoinCfwTest2(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [4, 5, 6], "idx": ["a", "b", "c"]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SecondCfw}


class JoinCfwTest3(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [7, 8, 9], "idx": ["a", "b", "c"]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {ThirdCfw}


class JoinCfwTest4(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [12, 13, 14], "idx": ["a", "b", "c"]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {FourthCfw}


############################################


class Join2CfwTest(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.int32_of("JoinCfwTest1"), Feature.int32_of("JoinCfwTest2")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data.column("JoinCfwTest1")
        data.column("JoinCfwTest2")
        return {cls.get_class_name(): [10, 2, 3]}


class Join3CfwTest(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.int32_of("JoinCfwTest1"), Feature.int32_of("JoinCfwTest2"), Feature.int32_of("JoinCfwTest3")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data.column("JoinCfwTest1")
        data.column("JoinCfwTest2")
        data.column("JoinCfwTest3")
        return {cls.get_class_name(): [33, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


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
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


class MultiIndexJoinTest1(FeatureGroup):
    """Feature group using Pandas framework with 2-column multi-index."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            cls.get_class_name(): [100, 200, 300],
            "id": [1, 2, 3],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class MultiIndexJoinTest2(FeatureGroup):
    """Feature group using PolarsLazy framework with 2-column multi-index."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            cls.get_class_name(): [101, 201, 301],
            "id": [1, 2, 3],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PolarsLazyDataFrame}


class MultiIndexJoinTest3(FeatureGroup):
    """Feature group using PyArrow framework with 2-column multi-index."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            cls.get_class_name(): [102, 202, 302],
            "id": [1, 2, 3],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


class MultiIndexJoinTest4(FeatureGroup):
    """Feature group using PythonDict framework with 2-column multi-index."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            cls.get_class_name(): [103, 203, 303],
            "id": [1, 2, 3],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}


class JoinMultiIndexTest(FeatureGroup):
    """Consumer feature group that uses all 4 multi-index feature groups."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature.int32_of("MultiIndexJoinTest1"),
            Feature.int32_of("MultiIndexJoinTest2"),
            Feature.int32_of("MultiIndexJoinTest3"),
            Feature.int32_of("MultiIndexJoinTest4"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Validate all columns are present
        data.column("MultiIndexJoinTest1")
        data.column("MultiIndexJoinTest2")
        data.column("MultiIndexJoinTest3")
        data.column("MultiIndexJoinTest4")
        return {cls.get_class_name(): [999, 888, 777]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


@PARALLELIZATION_MODES_SYNC_THREADING
class TestEngineMultipleJoinCfw:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def test_runner_join_multiple_cfw1_most_basic(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        def inner_loop(join_type: str) -> None:
            idx = Index(
                ("idx",),
            )

            left = JoinSpec(JoinCfwTest1, idx)
            right = JoinSpec(JoinCfwTest2, idx)

            links = {Link(join_type, left, right)}

            features = self.get_features(["Join2CfwTest"])
            runner = MlodaTestRunner.run_engine(
                features,
                compute_frameworks=COMPUTE_FRAMEWORKS,
                parallelization_modes=modes,
                flight_server=flight_server,
                links=links,
            )
            res = runner.get_result()
            assert res[0].to_pydict() == {"Join2CfwTest": [10, 2, 3]}

        for join_type in ["inner", "left", "right", "outer"]:
            inner_loop(join_type)

    def base_join_runner(
        self, modes: Set[ParallelizationMode], flight_server: Any, links: Set[Link], f_name: str = "3"
    ) -> None:
        f_name = f"Join{f_name}CfwTest"
        features = self.get_features([f_name])
        runner = MlodaTestRunner.run_engine(
            features,
            compute_frameworks=COMPUTE_FRAMEWORKS,
            parallelization_modes=modes,
            flight_server=flight_server,
            links=links,
        )
        res = runner.get_result()
        assert res[0].to_pydict() == {f_name: [33, 2, 3]}

    def test_runner_join_multiple_cfw2_join_to_same_base(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left = JoinSpec(JoinCfwTest1, idx)
        right_1 = JoinSpec(JoinCfwTest2, idx)
        right_2 = JoinSpec(JoinCfwTest3, idx)
        links = {Link("inner", left, right_1), Link("inner", left, right_2)}

        self.base_join_runner(modes, flight_server, links)

    def test_runner_join_multiple_cfw3_join_to_right_base(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left = JoinSpec(JoinCfwTest1, idx)
        right_1 = JoinSpec(JoinCfwTest2, idx)
        right_2 = JoinSpec(JoinCfwTest3, idx)
        links = {Link("inner", right_1, left), Link("inner", right_2, left)}

        self.base_join_runner(modes, flight_server, links)

    def test_runner_join_multiple_cfw4_chained_join(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        idx = Index(
            ("idx",),
        )

        left = JoinSpec(JoinCfwTest1, idx)
        right_1 = JoinSpec(JoinCfwTest2, idx)
        right_2 = JoinSpec(JoinCfwTest3, idx)
        links = {Link("inner", left, right_1), Link("inner", right_1, right_2)}

        self.base_join_runner(modes, flight_server, links)

    def test_runner_join_multiple_cfw5_double_chained_join(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left = JoinSpec(JoinCfwTest1, idx)
        right_1 = JoinSpec(JoinCfwTest2, idx)
        right_2 = JoinSpec(JoinCfwTest3, idx)
        right_3 = JoinSpec(JoinCfwTest4, idx)
        links = {Link("inner", left, right_1), Link("inner", right_1, right_2), Link("inner", right_2, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("inner", left, right_1), Link("inner", right_1, right_2), Link("inner", right_1, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

    def test_runner_join_multiple_cfw6_double_chained_join(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left = JoinSpec(JoinCfwTest1, idx)
        right_1 = JoinSpec(JoinCfwTest2, idx)
        right_2 = JoinSpec(JoinCfwTest3, idx)
        right_3 = JoinSpec(JoinCfwTest4, idx)
        links = {Link("inner", left, right_1), Link("inner", left, right_2), Link("inner", left, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("inner", left, right_1), Link("inner", right_2, left), Link("inner", left, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("inner", left, right_1), Link("inner", right_2, left), Link("inner", right_3, left)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

    def test_runner_join_multiple_cfw6_double_chained_join_left(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left = JoinSpec(JoinCfwTest1, idx)
        right_1 = JoinSpec(JoinCfwTest2, idx)
        right_2 = JoinSpec(JoinCfwTest3, idx)
        right_3 = JoinSpec(JoinCfwTest4, idx)

        links = {Link("left", left, right_1), Link("left", left, right_2), Link("left", left, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("left", left, right_1), Link("left", right_2, left), Link("left", left, right_3)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("outer", left, right_1), Link("inner", right_2, left), Link("inner", right_3, left)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

    def test_runner_join_multiple_cfw6_double_chained_join_right(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left = JoinSpec(JoinCfwTest1, idx)
        right_2 = JoinSpec(JoinCfwTest2, idx)
        right_3 = JoinSpec(JoinCfwTest3, idx)
        right_4 = JoinSpec(JoinCfwTest4, idx)

        links = {Link("left", left, right_2), Link("right", right_3, left), Link("left", left, right_4)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("inner", left, right_2), Link("right", left, right_3), Link("right", left, right_4)}
        with pytest.raises(ValueError):
            self.base_join_runner(modes, flight_server, links, f_name="4")

        with pytest.raises(ValueError):
            links = {Link("right", left, right_2), Link("inner", right_3, left), Link("right", right_4, left)}
            self.base_join_runner(modes, flight_server, links, f_name="4")

    def test_runner_join_multiple_cfw7_double_chained_join(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        idx = Index(
            ("idx",),
        )

        left = JoinSpec(JoinCfwTest1, idx)
        right_2 = JoinSpec(JoinCfwTest2, idx)
        right_3 = JoinSpec(JoinCfwTest3, idx)
        right_4 = JoinSpec(JoinCfwTest4, idx)

        links = {Link("inner", left, right_2), Link("inner", right_3, left), Link("inner", right_3, right_4)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

        links = {Link("inner", right_2, left), Link("inner", right_4, right_3), Link("inner", right_2, right_4)}
        self.base_join_runner(modes, flight_server, links, f_name="4")

    def test_raise_exception_if_duplicated_links_given(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        """This test should be moved to a unit test rather as it tests that Link.validate in engine init is working correctly. Was faster to test here."""

        idx = Index(
            ("idx",),
        )

        left = JoinSpec(JoinCfwTest1, idx)
        right_2 = JoinSpec(JoinCfwTest2, idx)
        right_3 = JoinSpec(JoinCfwTest3, idx)
        right_4 = JoinSpec(JoinCfwTest4, idx)

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
