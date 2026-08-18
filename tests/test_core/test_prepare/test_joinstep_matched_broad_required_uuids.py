"""An unrelated, unlinked parent must not skip the missing-Link conflict check by being wrongly
treated as "served by" a join it isn't genuinely a side of."""

from typing import Any, Optional

import pytest

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, Link, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class LinkedRootA(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"linked_root_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"linked_root_a": [1, 2, 3], "id": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("id",))]


class LinkedRootB(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"linked_root_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow as pa

        return pa.table({"linked_root_b": [10, 20, 30], "id": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("id",))]


class OrphanRootC(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"orphan_root_c"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"orphan_root_c": [100, 200, 300]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}


class OrphanRootD(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"orphan_root_d"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"orphan_root_d": [1000, 2000, 3000]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}


class FourParentConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature("linked_root_a"),
            Feature("linked_root_b"),
            Feature("orphan_root_c"),
            Feature("orphan_root_d"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"four_parent_consumer_result"}


class ThreeParentConsumer(FeatureGroup):
    """A join-served linked pair (zero explicit hops) plus an orphan (one explicit hop)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature("linked_root_a"),
            Feature("linked_root_b"),
            Feature("orphan_root_c"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"three_parent_consumer_result"}


def test_joinstep_matched_does_not_swallow_unrelated_unlinked_parent() -> None:
    with pytest.raises(ValueError) as exc_info:
        mloda.prepare(
            features=[Feature("four_parent_consumer_result")],
            links={Link.inner_on(LinkedRootA, LinkedRootB)},
            compute_frameworks={PandasDataFrame, PyArrowTable, PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {LinkedRootA, LinkedRootB, OrphanRootC, OrphanRootD, FourParentConsumer}
            ),
        )

    error_message = str(exc_info.value)

    assert "Link" in error_message, "Error should mention Links as the missing piece of guidance"
    assert "OrphanRootC" in error_message, "Error should name the first conflicting unlinked upstream feature group"
    assert "OrphanRootD" in error_message, "Error should name the second conflicting unlinked upstream feature group"


def test_joinstep_matched_raises_for_literal_three_parent_scenario() -> None:
    with pytest.raises(ValueError) as exc_info:
        mloda.prepare(
            features=[Feature("three_parent_consumer_result")],
            links={Link.inner_on(LinkedRootA, LinkedRootB)},
            compute_frameworks={PandasDataFrame, PyArrowTable, PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {LinkedRootA, LinkedRootB, OrphanRootC, ThreeParentConsumer}
            ),
        )

    error_message = str(exc_info.value)

    assert "Link" in error_message, "Error should mention Links as the missing piece of guidance"
    assert "OrphanRootC" in error_message, "Error should name the unlinked orphan feature group"
    assert "LinkedRootA" in error_message or "LinkedRootB" in error_message, (
        "Error should name the linked side it conflicts with"
    )


class IndependentJoinPairOneLeft(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"independent_join_pair_one_left"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"independent_join_pair_one_left": [1, 2, 3], "pair_one_id": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("pair_one_id",))]


class IndependentJoinPairOneRight(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"independent_join_pair_one_right"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"independent_join_pair_one_right": [10, 20, 30], "pair_one_id": [1, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("pair_one_id",))]


class IndependentJoinPairTwoLeft(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"independent_join_pair_two_left"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"independent_join_pair_two_left": [4, 5, 6], "pair_two_id": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("pair_two_id",))]


class IndependentJoinPairTwoRight(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"independent_join_pair_two_right"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"independent_join_pair_two_right": [40, 50, 60], "pair_two_id": [1, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("pair_two_id",))]


class TwoIndependentJoinsConsumer(FeatureGroup):
    """Both parents are join-served by two separate, unlinked JoinSteps: zero explicit hops at all."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature("independent_join_pair_one_left"),
            Feature("independent_join_pair_one_right"),
            Feature("independent_join_pair_two_left"),
            Feature("independent_join_pair_two_right"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"two_independent_joins_consumer_result"}


def test_joinstep_matched_raises_when_every_parent_is_join_served() -> None:
    with pytest.raises(ValueError) as exc_info:
        mloda.prepare(
            features=[Feature("two_independent_joins_consumer_result")],
            links={
                Link.inner_on(IndependentJoinPairOneLeft, IndependentJoinPairOneRight),
                Link.inner_on(IndependentJoinPairTwoLeft, IndependentJoinPairTwoRight),
            },
            compute_frameworks={PandasDataFrame, PyArrowTable, PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {
                    IndependentJoinPairOneLeft,
                    IndependentJoinPairOneRight,
                    IndependentJoinPairTwoLeft,
                    IndependentJoinPairTwoRight,
                    TwoIndependentJoinsConsumer,
                }
            ),
        )

    error_message = str(exc_info.value)

    assert "Link" in error_message, "Error should mention Links as the missing piece of guidance"
    # Either member of a pair can be the one a group reports first (both are join-served by the same
    # JoinStep and merge into one group), so accept either side's name for each of the two pairs.
    assert any(name in error_message for name in ("IndependentJoinPairOneLeft", "IndependentJoinPairOneRight")), (
        "Error should name a feature group from the first independently join-served pair"
    )
    assert any(name in error_message for name in ("IndependentJoinPairTwoLeft", "IndependentJoinPairTwoRight")), (
        "Error should name a feature group from the second independently join-served pair"
    )
