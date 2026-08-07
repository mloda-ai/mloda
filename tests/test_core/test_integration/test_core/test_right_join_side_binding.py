"""A RIGHT join binds the declared left group as the left merge argument; surplus left keys expose a swap."""

from typing import Any, Optional

import pytest

from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec, Link
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


LEFT_KEYS = [3, 2, 1]
LEFT_PAYLOADS = ["l3", "l2", "l1"]
RIGHT_KEYS = [1, 2]
RIGHT_PAYLOADS = ["r1", "r2"]

RIGHT_JOIN_ROWS = ["1|l1|1|r1", "2|l2|2|r2"]


class RightBindLeftInArrow(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"rjsb_left_key", "rjsb_left_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rjsb_left_key": LEFT_KEYS, "rjsb_left_payload": LEFT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class RightBindRightInPandas(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"rjsb_right_key", "rjsb_right_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rjsb_right_key": RIGHT_KEYS, "rjsb_right_payload": RIGHT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class RightBindChild(FeatureGroup):
    """Runs in the declared right group's framework, which is where the RIGHT join executes."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature(name="rjsb_left_key"),
            Feature(name="rjsb_left_payload"),
            Feature(name="rjsb_right_key"),
            Feature(name="rjsb_right_payload"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        rows = [
            "|".join(str(value) for value in row)
            for row in zip(
                data["rjsb_left_key"],
                data["rjsb_left_payload"],
                data["rjsb_right_key"],
                data["rjsb_right_payload"],
            )
        ]
        return {cls.get_class_name(): rows}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


# MULTIPROCESSING is left out: cross-framework joins fail in the transform hop for unrelated reasons.
@pytest.mark.parametrize("modes", [{ParallelizationMode.SYNC}, {ParallelizationMode.THREADING}])
def test_right_join_keeps_every_right_row_and_drops_unmatched_left_rows(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    link = Link.right(
        JoinSpec(RightBindLeftInArrow, Index(("rjsb_left_key",))),
        JoinSpec(RightBindRightInPandas, Index(("rjsb_right_key",))),
    )

    results = mloda.run_all(
        [Feature(name=RightBindChild.get_class_name())],
        links={link},
        compute_frameworks={PandasDataFrame, PyArrowTable},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {RightBindLeftInArrow, RightBindRightInPandas, RightBindChild}
        ),
        flight_server=flight_server if ParallelizationMode.MULTIPROCESSING in modes else None,
        parallelization_modes=modes,
    )

    joined = [frame for frame in results if RightBindChild.get_class_name() in list(frame.columns)]
    assert len(joined) == 1
    rows = list(joined[0][RightBindChild.get_class_name()])

    assert len(rows) == len(RIGHT_KEYS)
    assert sorted(rows) == RIGHT_JOIN_ROWS
