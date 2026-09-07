"""Sibling feature groups on one parent resolve to a single compute framework instance.
Both outputs must survive when the two steps overlap in time.
"""

import threading
from typing import Any

import pytest

from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

HANDSHAKE_TIMEOUT = 1.0

late_reader_took_input = threading.Event()
early_writer_stored_data = threading.Event()
late_writer_stored_data = threading.Event()


class SharedSiblingParent(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SiblingEarlyWriter(FeatureGroup):
    """Stores its frame first, then holds its step open until the sibling has stored too."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name=SharedSiblingParent.get_class_name())}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        late_reader_took_input.wait(HANDSHAKE_TIMEOUT)
        return data.assign(**{cls.get_class_name(): data[SharedSiblingParent.get_class_name()]})

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        early_writer_stored_data.set()
        late_writer_stored_data.wait(HANDSHAKE_TIMEOUT)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SiblingLateWriter(FeatureGroup):
    """Takes its input before the sibling writes, then stores its own frame last."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name=SharedSiblingParent.get_class_name())}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        late_reader_took_input.set()
        early_writer_stored_data.wait(HANDSHAKE_TIMEOUT)
        return data.assign(**{cls.get_class_name(): data[SharedSiblingParent.get_class_name()]})

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        late_writer_stored_data.set()

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


@pytest.fixture(autouse=True)
def reset_handshake() -> None:
    late_reader_took_input.clear()
    early_writer_stored_data.clear()
    late_writer_stored_data.clear()


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationMode.SYNC}),
        ({ParallelizationMode.THREADING}),
    ],
)
class TestSiblingStepsShareOneCfw:
    def test_both_sibling_outputs_survive(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        result = mloda.run_all(
            [
                Feature(name=SiblingEarlyWriter.get_class_name()),
                Feature(name=SiblingLateWriter.get_class_name()),
            ],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {SharedSiblingParent, SiblingEarlyWriter, SiblingLateWriter}
            ),
            flight_server=flight_server,
            parallelization_modes=modes,
        )

        seen: set[str] = set()
        for res in result:
            seen.update(res.columns)

        assert SiblingEarlyWriter.get_class_name() in seen
        assert SiblingLateWriter.get_class_name() in seen
