"""An APPEND consumer resolving to a different framework than the left index feature is a configuration error.

A second consumer sharing the same link does not change that outcome.
"""

from typing import Any

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


# The message must name the orientation concept and say that it is not supported.
ORIENTATION_WORDS = ("invert", "revers", "swap", "orientation")
UNSUPPORTED_WORDS = ("not support", "unsupported", "cannot", "can't", "not allowed")


def _assert_is_orientation_configuration_error(message: str, link: Link) -> None:
    """The error reads as a configuration problem, not an internal bug report."""
    assert not message.startswith("Internal error:")
    assert "report this issue" not in message.lower()
    assert "sanity check" not in message
    assert str(link) in message
    assert PyArrowTable.get_class_name() in message
    assert PandasDataFrame.get_class_name() in message
    assert link.jointype.value in message.lower()
    assert any(word in message.lower() for word in ORIENTATION_WORDS)
    assert any(word in message.lower() for word in UNSUPPORTED_WORDS)


class SharedAppendSource(FeatureGroup):
    """Serves both sides of the append; each side is pinned to a different framework."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(
            supports_features={"stack_left_key", "stack_left_payload", "stack_right_key", "stack_right_payload"}
        )

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        names = {str(feature.name) for feature in features.features}
        if names & {"stack_left_key", "stack_left_payload"}:
            return {"stack_left_key": [1, 2], "stack_left_payload": ["l1", "l2"]}
        return {"stack_right_key": [3, 4], "stack_right_payload": ["r3", "r4"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable, PandasDataFrame}


def _append_sides() -> set[Feature]:
    return {
        Feature(name="stack_left_payload", compute_framework="PyArrowTable", index=Index(("stack_left_key",))),
        Feature(name="stack_right_payload", compute_framework="PandasDataFrame", index=Index(("stack_right_key",))),
    }


class SharedAppendPandasConsumer(FeatureGroup):
    """Resolves to the right framework only, so it differs from the left index feature."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return _append_sides()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = data["stack_left_payload"]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SharedAppendFlexibleConsumer(FeatureGroup):
    """Second consumer of the same link, kept unpinned so only the link is shared."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return _append_sides()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = data["stack_left_payload"]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable, PandasDataFrame}


def _append_link() -> Link:
    return Link.append(
        left=JoinSpec(SharedAppendSource, Index(("stack_left_key",))),
        right=JoinSpec(SharedAppendSource, Index(("stack_right_key",))),
    )


def _plan_append(link: Link, consumers: list[type[FeatureGroup]]) -> None:
    mloda.run_all(
        [Feature(name=consumer.get_class_name()) for consumer in consumers],
        links={link},
        compute_frameworks=["PyArrowTable", "PandasDataFrame"],
        plugin_collector=PluginCollector.enabled_feature_groups({SharedAppendSource, *consumers}),
        parallelization_modes={ParallelizationMode.SYNC},
    )


def test_consumer_framework_mismatch_is_rejected_while_building_the_joinstep() -> None:
    link = _append_link()

    with pytest.raises(ValueError) as excinfo:
        _plan_append(link, [SharedAppendPandasConsumer])

    _assert_is_orientation_configuration_error(str(excinfo.value), link)


def test_consumer_framework_mismatch_names_the_link_and_both_frameworks() -> None:
    link = _append_link()

    with pytest.raises(ValueError) as excinfo:
        _plan_append(link, [SharedAppendPandasConsumer])

    _assert_is_orientation_configuration_error(str(excinfo.value), link)


def test_a_second_consumer_of_the_same_link_raises_the_same_error() -> None:
    link = _append_link()

    with pytest.raises(ValueError) as excinfo:
        _plan_append(link, [SharedAppendPandasConsumer, SharedAppendFlexibleConsumer])

    _assert_is_orientation_configuration_error(str(excinfo.value), link)
