"""Pin BaseArtifact.custom_loader to key off artifact_to_load, not name_of_one_feature (always the
alphabetically smallest feature name, independent of which feature the artifact is stored under)."""

from mloda.provider import BaseArtifact, FeatureSet
from mloda.user import Feature, Options


class TestBaseArtifactCustomLoaderUsesArtifactToLoad:
    def test_loads_stored_value_when_artifact_key_is_not_alphabetically_smallest(self) -> None:
        shared_options = Options({"z_col": "stored_value"})
        features = FeatureSet()
        features.add(Feature("a_col", shared_options))
        features.add(Feature("z_col", shared_options))

        features.add_artifact_name()

        # Sanity on the two attributes the bug conflates.
        assert features.artifact_to_load == "z_col"
        assert features.name_of_one_feature == "a_col"

        result = BaseArtifact.custom_loader(features)

        assert result == "stored_value"

    def test_load_uses_artifact_to_load_when_artifact_key_is_not_alphabetically_smallest(self) -> None:
        shared_options = Options({"z_col": "stored_value"})
        features = FeatureSet()
        features.add(Feature("a_col", shared_options))
        features.add(Feature("z_col", shared_options))

        features.add_artifact_name()

        result = BaseArtifact.load(features)

        assert result == "stored_value"


class TestFeatureSetResolveArtifactForRuntimeUsesArtifactToLoad:
    """resolve_artifact_for_runtime() is the second producer of artifact_to_load, used on the
    prepare() + run(artifacts=...) path; it must feed custom_loader/load the same way add_artifact_name() does."""

    def test_loads_runtime_value_when_artifact_key_is_not_alphabetically_smallest(self) -> None:
        shared_options = Options({})
        features = FeatureSet()
        features.add(Feature("a_col", shared_options))
        features.add(Feature("z_col", shared_options))

        features.resolve_artifact_for_runtime({"z_col": "runtime_value"})

        # Sanity on the two attributes the bug conflates.
        assert features.artifact_to_load == "z_col"
        assert features.name_of_one_feature == "a_col"

        result = BaseArtifact.custom_loader(features)

        assert result == "runtime_value"

    def test_load_uses_artifact_to_load_when_resolved_at_runtime(self) -> None:
        shared_options = Options({})
        features = FeatureSet()
        features.add(Feature("a_col", shared_options))
        features.add(Feature("z_col", shared_options))

        features.resolve_artifact_for_runtime({"z_col": "runtime_value"})

        result = BaseArtifact.load(features)

        assert result == "runtime_value"

    def test_runtime_artifact_does_not_change_feature_identity_or_group_options(self) -> None:
        shared_options = Options(group={"model_version": "v1"}, context={"trace_id": "run-1"})
        feature = Feature("artifact_feature", shared_options)
        features = FeatureSet([feature])
        feature_hash = hash(feature)
        group_options = shared_options.group.copy()

        features.resolve_artifact_for_runtime({"artifact_feature": "runtime_value"})

        assert features.artifact_to_load == "artifact_feature"
        assert features.artifact_to_save is None
        assert feature in features.features
        assert hash(feature) == feature_hash
        assert shared_options.group == group_options
        assert BaseArtifact.load(features) == "runtime_value"
        assert feature in features.features
        assert hash(feature) == feature_hash
        assert shared_options.group == group_options
