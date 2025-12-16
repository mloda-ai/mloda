from unittest.mock import patch

import pytest
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import (
    BaseTestComputeFramework2,
    BaseTestComputeFramework1,
)


from tests.test_core.test_abstract_plugins.test_abstract_feature_group import (
    BaseTestFeatureGroup1,
    BaseTestFeatureGroup2,
)


class TestAccessibleCollecton:
    def test_accessible_plugins(self) -> None:
        with (
            patch(
                "mloda.core.prepare.accessible_plugins.PreFilterPlugins.get_featuregroup_subclasses"
            ) as mocked_accessible_feature_groups,
            patch(
                "mloda.core.prepare.accessible_plugins.PreFilterPlugins.get_cfw_subclasses"
            ) as mocked_accessible_environments,
        ):
            mocked_accessible_feature_groups.return_value = {
                BaseTestFeatureGroup1,
                BaseTestFeatureGroup2,
            }

            fw = {BaseTestComputeFramework1, BaseTestComputeFramework2}
            mocked_accessible_environments.return_value = fw

            accessible_plugins = PreFilterPlugins(fw)

            assert len(accessible_plugins._set_feature_groups()) == 2

            assert len(accessible_plugins._set_compute_frameworks(set())) == 0
            assert len(accessible_plugins._set_compute_frameworks(fw)) == 2

            mapping = accessible_plugins.get_accessible_plugins()
            assert mapping.keys() == {BaseTestFeatureGroup1, BaseTestFeatureGroup2}

            for _, v in mapping.items():
                assert v == {BaseTestComputeFramework1, BaseTestComputeFramework2}

    def test_accessible_feature_groups_empty(self) -> None:
        with (
            patch(
                "mloda.core.prepare.accessible_plugins.PreFilterPlugins.get_featuregroup_subclasses"
            ) as mocked_accessible_feature_groups,
            patch(
                "mloda.core.prepare.accessible_plugins.PreFilterPlugins.get_cfw_subclasses"
            ) as mocked_accessible_environments,
        ):
            mocked_accessible_feature_groups.return_value = set()

            fw = {BaseTestComputeFramework1, BaseTestComputeFramework2}
            mocked_accessible_environments.return_value = fw

            with pytest.raises(ValueError):
                PreFilterPlugins(fw)
