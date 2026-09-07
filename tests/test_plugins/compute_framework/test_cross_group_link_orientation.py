"""Two child groups sharing one link between the same parent pair.

Both must end up on a single orientation and see the joined columns.
"""

from typing import Any

import pyarrow as pa
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


class OrientParentLeft(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["Same Value"], "orient_left_payload": ["left"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class OrientParentRight(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["Same Value"], "orient_right_payload": ["right"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


def _column_names(data: Any) -> list[str]:
    if isinstance(data, pa.Table):
        return list(data.column_names)
    return list(data.columns)


class OrientChildFlexible(FeatureGroup):
    """Supports both frameworks, so on its own it would keep the left orientation."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name=OrientParentLeft.get_class_name()), Feature(name=OrientParentRight.get_class_name())}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        columns = _column_names(data)
        for expected in ("orient_left_payload", "orient_right_payload"):
            if expected not in columns:
                raise ValueError(f"{expected} not in data: {columns}")

        if isinstance(data, pa.Table):
            return data.append_column(cls.get_class_name(), data[OrientParentLeft.get_class_name()])

        data[cls.get_class_name()] = data[OrientParentLeft.get_class_name()]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable, PandasDataFrame}


class OrientChildPandasOnly(FeatureGroup):
    """Supports only the right framework, so on its own it would invert the link."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name=OrientParentLeft.get_class_name()), Feature(name=OrientParentRight.get_class_name())}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        columns = _column_names(data)
        for expected in ("orient_left_payload", "orient_right_payload"):
            if expected not in columns:
                raise ValueError(f"{expected} not in data: {columns}")

        data[cls.get_class_name()] = data[OrientParentLeft.get_class_name()]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationMode.SYNC}),
        ({ParallelizationMode.THREADING}),
    ],
)
class TestCrossGroupLinkOrientation:
    def test_shared_link_serves_both_child_groups(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        link = Link.inner(
            left=JoinSpec(OrientParentLeft, Index((OrientParentLeft.get_class_name(),))),
            right=JoinSpec(OrientParentRight, Index((OrientParentRight.get_class_name(),))),
        )

        result = mloda.run_all(
            [
                Feature(name=OrientChildFlexible.get_class_name()),
                Feature(name=OrientChildPandasOnly.get_class_name()),
            ],
            links={link},
            compute_frameworks=["PyArrowTable", "PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {
                    OrientParentLeft,
                    OrientParentRight,
                    OrientChildFlexible,
                    OrientChildPandasOnly,
                }
            ),
            flight_server=flight_server,
            parallelization_modes=modes,
        )

        seen: set[str] = set()
        for res in result:
            seen.update(_column_names(res))

        assert OrientChildFlexible.get_class_name() in seen
        assert OrientChildPandasOnly.get_class_name() in seen
