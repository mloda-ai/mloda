"""Tests for improved error messages in TransformFrameworkStep."""

from typing import Optional
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep


class MockFromFramework(ComputeFramework):
    pass


class MockToFramework(ComputeFramework):
    pass


class MockFeatureGroup(FeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestTransformFrameworkStepFromCfwNoneError:
    """Tests that from_cfw=None produces an actionable error message."""

    def _make_step(self) -> TransformFrameworkStep:
        return TransformFrameworkStep(
            from_framework=MockFromFramework,
            to_framework=MockToFramework,
            required_uuids={uuid4()},
            from_feature_group=MockFeatureGroup,
            to_feature_group=MockFeatureGroup,
        )

    def test_error_mentions_internal_error(self) -> None:
        step = self._make_step()
        cfw_register = MagicMock()
        cfw_register.get_location.return_value = None
        cfw = MagicMock(spec=ComputeFramework)

        with pytest.raises(ValueError, match="Internal error"):
            step.execute(cfw_register, cfw, from_cfw=None)

    def test_error_mentions_from_cfw(self) -> None:
        step = self._make_step()
        cfw_register = MagicMock()
        cfw_register.get_location.return_value = None
        cfw = MagicMock(spec=ComputeFramework)

        with pytest.raises(ValueError, match="from_cfw is None"):
            step.execute(cfw_register, cfw, from_cfw=None)

    def test_error_contains_report_url(self) -> None:
        step = self._make_step()
        cfw_register = MagicMock()
        cfw_register.get_location.return_value = None
        cfw = MagicMock(spec=ComputeFramework)

        with pytest.raises(ValueError, match="mloda-ai/mloda/issues"):
            step.execute(cfw_register, cfw, from_cfw=None)
