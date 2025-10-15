import json
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.config.feature.loader import load_features_from_config


def test_end2end_feature_config() -> None:
    """Test that feature config loader correctly parses config and creates Feature objects."""
    config_str = json.dumps(["age", {"name": "weight", "options": {"imputation_method": "mean"}}])

    features = load_features_from_config(config_str, format="json")

    # Verify we got 2 features
    assert len(features) == 2

    # First feature: simple string "age"
    assert features[0] == "age"

    # Second feature: Feature object with name and options
    assert isinstance(features[1], Feature)
    assert features[1].name.name == "weight"
    assert features[1].options.get("imputation_method") == "mean"
