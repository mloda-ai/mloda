"""End-to-end one-to-many inner join through run_all on PythonDict.

Proves the plumbing run_all -> JoinStep -> PythonDictMergeEngine.merge_inner
preserves one-to-many fan-out: a unique-key left joined to a duplicate-key right
must yield one row per matching right row, not collapse to the last one.
"""

from typing import Any, Optional

import pytest

from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec
from mloda.user import Link
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda

# Import to register PythonDictFramework as a discoverable ComputeFramework subclass.
import mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework  # noqa: F401


def _one_to_many_link() -> Link:
    """Inner join on user_id: unique-key left to duplicate-key right."""
    return Link(
        "inner",
        JoinSpec(OneToManyLeftFeature, Index(("user_id",))),
        JoinSpec(OneToManyRightFeature, Index(("user_id",))),
    )


class OneToManyLeftFeature(FeatureGroup):
    """Left side: one row per user (unique join keys)."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"user_id", "uname"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"user_id": [1, 2], "uname": ["ann", "bob"]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"user_id", "uname"}


class OneToManyRightFeature(FeatureGroup):
    """Right side: many rows per user (duplicate join keys)."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"user_id", "amount"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"user_id": [1, 1, 2, 2], "amount": [10, 20, 30, 40]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"user_id", "amount"}


class OneToManyJoinedFeature(FeatureGroup):
    """Parent consuming the one-to-many join; encodes each joined row as 'uname|amount'."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        link = _one_to_many_link()
        return {
            Feature(name="uname", link=link, index=Index(("user_id",))),
            Feature(name="amount", index=Index(("user_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        length = len(next(iter(data.values())))
        encoded = [f"{data['uname'][i]}|{data['amount'][i]}" for i in range(length)]
        return {cls.get_class_name(): encoded}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_ENABLED = PluginCollector.enabled_feature_groups({OneToManyLeftFeature, OneToManyRightFeature, OneToManyJoinedFeature})


class TestPythonDictOneToManyJoinRunAll:
    """A one-to-many inner join must fan out end-to-end through run_all."""

    @pytest.mark.parametrize("modes", [{ParallelizationMode.SYNC}, {ParallelizationMode.THREADING}])
    def test_one_to_many_inner_join(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """Unique-key left joined to a duplicate-key right yields four rows, not two."""
        feature = Feature(name=OneToManyJoinedFeature.get_class_name())
        result = mloda.run_all(
            [feature],
            links={_one_to_many_link()},
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            flight_server=flight_server,
            parallelization_modes=modes,
        )

        assert len(result) == 1
        column = result[0][OneToManyJoinedFeature.get_class_name()]
        assert sorted(column) == ["ann|10", "ann|20", "bob|30", "bob|40"]
