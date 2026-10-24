"""FeatureGroupStep execute behavior."""

from unittest.mock import MagicMock

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.user import Feature, Options


class MockFeatureGroup(FeatureGroup):
    """A mock feature group for testing error messages."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


def test_execute_does_not_upload_finished_data_twice() -> None:
    features = MagicMock(spec=FeatureSet)
    features.features = set()
    features.artifact_to_save = None
    step = FeatureGroupStep(MockFeatureGroup, features, set(), ComputeFramework)
    step.need_to_upload = True

    cfw_register = MagicMock()
    cfw_register.get_location.return_value = "flight-location"
    cfw_register.get_runtime_artifacts.return_value = None
    cfw = MagicMock(spec=ComputeFramework)
    cfw.uuid = "framework-id"

    def run_calculation(*_: object) -> str:
        cfw.data = "object-id"
        cfw.upload_finished_data("flight-location")
        return "object-id"

    cfw.run_calculation.side_effect = run_calculation

    step.execute(cfw_register, cfw)

    cfw.upload_finished_data.assert_called_once_with("flight-location")
