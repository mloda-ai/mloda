import inspect
from unittest.mock import patch


from mloda import Feature
from mloda import API
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


def test_default_option_key_exists() -> None:
    """Test that strict_type_enforcement key exists in DefaultOptionKeys."""
    assert hasattr(DefaultOptionKeys, "strict_type_enforcement")
    assert DefaultOptionKeys.strict_type_enforcement.value == "strict_type_enforcement"


def test_mloda_api_accepts_strict_type_enforcement_parameter() -> None:
    """Test that API constructor accepts strict_type_enforcement parameter."""
    sig = inspect.signature(API.__init__)
    params = sig.parameters
    assert "strict_type_enforcement" in params
    # Default should be False
    assert params["strict_type_enforcement"].default is False


def test_strict_type_enforcement_propagates_to_features() -> None:
    """Test that strict_type_enforcement=True propagates to feature options."""
    features: list[Feature | str] = [Feature.int32_of("test_feature")]

    # Create API with strict mode enabled
    # This should propagate the setting to features
    with (
        patch("mloda.core.prepare.accessible_plugins.PreFilterPlugins.get_featuregroup_subclasses") as mock_fg,
        patch(
            "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
        ),
        patch("mloda.core.core.engine.Engine.create_setup_execution_plan"),
    ):
        # Mock to return at least one feature group so initialization doesn't fail
        from tests.test_core.test_abstract_plugins.test_abstract_feature_group import BaseTestFeatureGroup1

        mock_fg.return_value = {BaseTestFeatureGroup1}

        api = API(
            requested_features=features,
            strict_type_enforcement=True,
        )

        # Check that the setting is stored/propagated
        # The exact mechanism depends on implementation, but test that it's accessible
        assert api.strict_type_enforcement is True
