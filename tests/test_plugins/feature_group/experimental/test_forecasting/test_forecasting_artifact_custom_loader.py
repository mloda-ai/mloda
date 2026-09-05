"""Pin ForecastingArtifact.custom_loader to key off artifact_to_load, not name_of_one_feature.
Same bug pattern as BaseArtifact.custom_loader."""

from mloda.provider import FeatureSet
from mloda.user import Feature, Options
from mloda_plugins.feature_group.experimental.forecasting.forecasting_artifact import ForecastingArtifact


class TestForecastingArtifactCustomLoaderUsesArtifactToLoad:
    def test_loads_stored_value_when_artifact_key_is_not_alphabetically_smallest(self) -> None:
        serialized = ForecastingArtifact._serialize_artifact({"feature_names": ["sales"]})
        shared_options = Options({"z_col": serialized})
        features = FeatureSet()
        features.add(Feature("a_col", shared_options))
        features.add(Feature("z_col", shared_options))

        features.add_artifact_name()

        # Sanity on the two attributes the bug conflates.
        assert features.artifact_to_load == "z_col"
        assert features.name_of_one_feature == "a_col"

        result = ForecastingArtifact.custom_loader(features)

        assert result is not None
        assert result["feature_names"] == ["sales"]

    def test_load_uses_artifact_to_load_when_artifact_key_is_not_alphabetically_smallest(self) -> None:
        serialized = ForecastingArtifact._serialize_artifact({"feature_names": ["sales"]})
        shared_options = Options({"z_col": serialized})
        features = FeatureSet()
        features.add(Feature("a_col", shared_options))
        features.add(Feature("z_col", shared_options))

        features.add_artifact_name()

        result = ForecastingArtifact.load(features)

        assert result is not None
        assert result["feature_names"] == ["sales"]
