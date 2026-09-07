"""Pin BaseArtifact.custom_loader to key off artifact_to_load, not name_of_one_feature (always the
alphabetically smallest feature name, independent of which feature the artifact is stored under)."""

import pytest

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


class TestFeatureSetResolveArtifactForRuntimeDoesNotMutateFeatureHash:
    """resolve_artifact_for_runtime() must not change the group options (and therefore the
    hash) of any Feature whose `.options` aliases the FeatureSet's shared Options object --
    doing so silently corrupts `self.features` set membership (#1236)."""

    def test_feature_remains_a_member_after_resolving_runtime_artifact(self) -> None:
        feature = Feature("abc")
        features = FeatureSet()
        features.add(feature)

        assert feature in features.features

        features.resolve_artifact_for_runtime({"abc": "ART"})

        assert feature in features.features

    def test_feature_group_options_are_unchanged_after_resolving_runtime_artifact(self) -> None:
        feature = Feature("abc")
        features = FeatureSet()
        features.add(feature)
        hash_before = hash(feature)

        features.resolve_artifact_for_runtime({"abc": "ART"})

        assert "abc" not in feature.options.group
        assert hash(feature) == hash_before

    def test_shared_options_group_is_unchanged_across_features(self) -> None:
        shared_options = Options({})
        features = FeatureSet()
        a_feature = Feature("a_col", shared_options)
        z_feature = Feature("z_col", shared_options)
        features.add(a_feature)
        features.add(z_feature)

        features.resolve_artifact_for_runtime({"z_col": "runtime_value"})

        assert "z_col" not in shared_options.group
        assert a_feature in features.features
        assert z_feature in features.features

    def test_artifact_is_still_retrievable_via_feature_options_after_fix(self) -> None:
        shared_options = Options({})
        features = FeatureSet()
        features.add(Feature("a_col", shared_options))
        features.add(Feature("z_col", shared_options))

        features.resolve_artifact_for_runtime({"z_col": "runtime_value"})

        assert shared_options["z_col"] == "runtime_value"


class TestFeatureSetResolveArtifactForRuntimeRejectsGroupCollision:
    """A stored artifact key baked into Options.group (the documented "load via Options" pattern)
    must not silently coexist with a runtime override written into context: Options.get() checks
    group before context, so the stale group value would shadow the fresh runtime value forever."""

    def test_raises_when_runtime_override_collides_with_group_baked_artifact(self) -> None:
        options = Options({"z_col": "stored_value"})
        feature = Feature("z_col", options)
        features = FeatureSet()
        features.add(feature)

        with pytest.raises(ValueError, match="(?i)artifact"):
            features.resolve_artifact_for_runtime({"z_col": "new_runtime_value"})


class TestFeatureSetResolveArtifactForRuntimeVisibleAcrossDistinctOptionsInstances:
    """resolve_artifact_for_runtime() writes only onto self.options (aliased to whichever Feature
    was added first). When a FeatureSet holds two value-equal but distinct Options instances, the
    runtime value must land on both, since get_singular_option_from_options reads an arbitrary
    feature from a set."""

    def test_runtime_value_lands_on_both_distinct_options_instances(self) -> None:
        options_a = Options({})
        options_z = Options({})
        a_col = Feature("a_col", options_a)
        z_col = Feature("z_col", options_z)
        fs = FeatureSet()
        fs.add(a_col)
        fs.add(z_col)

        fs.resolve_artifact_for_runtime({"z_col": "runtime_value"})

        assert options_a.context.get("z_col") == "runtime_value"
        assert options_z.context.get("z_col") == "runtime_value"
