"""Tests for domain propagation in engine.py's _handle_input_features_recursion method.

These tests verify that the parent feature's domain is correctly propagated
to child features during the recursion process in the Engine class.
"""

from unittest.mock import patch, MagicMock, call
from uuid import uuid4

import pytest

from mloda.core.core.engine import Engine
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.domain import Domain

from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import (
    BaseTestComputeFramework1,
    BaseTestComputeFramework2,
)
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import (
    BaseTestFeatureGroup1,
)


class TestEngineDomainPropagation:
    """Tests for domain propagation during input features recursion."""

    def test_handle_input_features_recursion_passes_domain(self) -> None:
        """Test that _handle_input_features_recursion passes parent_domain to Features constructor.

        When _handle_input_features_recursion is called with parent_domain="TestDomain",
        the Features constructor should receive parent_domain="TestDomain".

        This test should FAIL because:
        - Current method signature: _handle_input_features_recursion(self, feature_group_class, uuid, options, feature_name)
        - Expected method signature: _handle_input_features_recursion(self, feature_group_class, uuid, options, feature_name, parent_domain)
        """
        with (
            patch(
                "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
            ) as mocked_derived_accessible_plugins,
            patch("mloda.core.core.engine.Engine.create_setup_execution_plan"),
        ):
            mocked_derived_accessible_plugins.return_value = {
                BaseTestFeatureGroup1: [BaseTestComputeFramework1, BaseTestComputeFramework2],
            }

            features = Features(["BaseTestFeature1"])
            compute_framework = {BaseTestComputeFramework1, BaseTestComputeFramework2}
            engine = Engine(features, compute_framework, None)

            test_uuid = uuid4()
            test_options = Options({})
            test_feature_name = FeatureName("TestFeature")
            test_domain = "TestDomain"

            mock_feature_group = MagicMock()
            mock_feature_group.input_features.return_value = [Feature(name="ChildFeature")]

            with patch.object(BaseTestFeatureGroup1, "__new__", return_value=mock_feature_group):
                with patch("mloda.core.core.engine.Features") as MockFeatures:
                    MockFeatures.return_value = MagicMock(child_uuid=test_uuid, parent_uuids=set())

                    engine._handle_input_features_recursion(
                        BaseTestFeatureGroup1,
                        test_uuid,
                        test_options,
                        test_feature_name,
                        parent_domain=test_domain,
                    )

                    MockFeatures.assert_called_once()
                    call_kwargs = MockFeatures.call_args
                    assert call_kwargs.kwargs.get("parent_domain") == test_domain, (
                        f"Expected parent_domain='TestDomain' in Features constructor call, but got: {call_kwargs}"
                    )

    def test_process_feature_extracts_domain_for_recursion(self) -> None:
        """Test that _process_feature extracts domain from feature and passes to recursion.

        When a feature has a domain set (e.g., feature.domain.name = "TestDomain"),
        the _process_feature method should extract this domain and pass it to
        _handle_input_features_recursion as parent_domain.

        This test should FAIL because:
        - Current call: self._handle_input_features_recursion(feature_group_class, feature.uuid, feature.options, feature.name)
        - Expected call: self._handle_input_features_recursion(feature_group_class, feature.uuid, feature.options, feature.name, parent_domain=feature.domain.name)
        """
        with (
            patch(
                "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
            ) as mocked_derived_accessible_plugins,
            patch("mloda.core.core.engine.Engine.create_setup_execution_plan"),
        ):
            mocked_derived_accessible_plugins.return_value = {
                BaseTestFeatureGroup1: [BaseTestComputeFramework1, BaseTestComputeFramework2],
            }

            features = Features(["BaseTestFeature1"])
            compute_framework = {BaseTestComputeFramework1, BaseTestComputeFramework2}
            engine = Engine(features, compute_framework, None)

            test_feature = Feature(name="TestFeature", domain="TestDomain")
            test_feature.compute_frameworks = {BaseTestComputeFramework1}

            mock_features = MagicMock()
            mock_features.child_uuid = None

            with (
                patch.object(engine, "_identify_feature_group_and_frameworks") as mock_identify,
                patch.object(engine, "add_feature_to_collection", return_value=True),
                patch.object(engine, "_handle_input_features_recursion") as mock_recursion,
            ):
                mock_identify.return_value = (BaseTestFeatureGroup1, {BaseTestComputeFramework1})

                engine._process_feature(test_feature, mock_features)

                mock_recursion.assert_called_once()
                call_args = mock_recursion.call_args

                assert "parent_domain" in call_args.kwargs, (
                    f"Expected parent_domain keyword argument in _handle_input_features_recursion call, "
                    f"but got: args={call_args.args}, kwargs={call_args.kwargs}"
                )
                assert call_args.kwargs["parent_domain"] == "TestDomain", (
                    f"Expected parent_domain='TestDomain', but got: {call_args.kwargs.get('parent_domain')}"
                )

    def test_process_feature_passes_none_domain_when_feature_has_no_domain(self) -> None:
        """Test that _process_feature passes None for parent_domain when feature has no domain.

        When a feature does not have a domain set (feature.domain is None),
        the _process_feature method should pass parent_domain=None to
        _handle_input_features_recursion.

        This test should FAIL because:
        - Current implementation doesn't pass parent_domain parameter at all
        """
        with (
            patch(
                "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
            ) as mocked_derived_accessible_plugins,
            patch("mloda.core.core.engine.Engine.create_setup_execution_plan"),
        ):
            mocked_derived_accessible_plugins.return_value = {
                BaseTestFeatureGroup1: [BaseTestComputeFramework1, BaseTestComputeFramework2],
            }

            features = Features(["BaseTestFeature1"])
            compute_framework = {BaseTestComputeFramework1, BaseTestComputeFramework2}
            engine = Engine(features, compute_framework, None)

            test_feature = Feature(name="TestFeature")
            test_feature.compute_frameworks = {BaseTestComputeFramework1}
            assert test_feature.domain is None

            mock_features = MagicMock()
            mock_features.child_uuid = None

            with (
                patch.object(engine, "_identify_feature_group_and_frameworks") as mock_identify,
                patch.object(engine, "add_feature_to_collection", return_value=True),
                patch.object(engine, "_handle_input_features_recursion") as mock_recursion,
            ):
                mock_identify.return_value = (BaseTestFeatureGroup1, {BaseTestComputeFramework1})

                engine._process_feature(test_feature, mock_features)

                mock_recursion.assert_called_once()
                call_args = mock_recursion.call_args

                assert "parent_domain" in call_args.kwargs, (
                    f"Expected parent_domain keyword argument in _handle_input_features_recursion call, "
                    f"but got: args={call_args.args}, kwargs={call_args.kwargs}"
                )
                assert call_args.kwargs["parent_domain"] is None, (
                    f"Expected parent_domain=None for feature without domain, "
                    f"but got: {call_args.kwargs.get('parent_domain')}"
                )
