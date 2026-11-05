"""Pin BaseArtifact.custom_loader to key off artifact_to_load, not name_of_one_feature (always the
alphabetically smallest feature name, independent of which feature the artifact is stored under)."""

from typing import Any

import numpy as np
import pytest

from mloda.core.abstract_plugins.components.feature_set import ArtifactBinding
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


def _run_view(features: FeatureSet, artifacts: dict[str, object]) -> FeatureSet:
    return features.for_run(features.runtime_artifact_binding(artifacts))


def _loaded_array(view: FeatureSet) -> np.ndarray[Any, Any]:
    loaded = BaseArtifact.load(view)
    assert isinstance(loaded, np.ndarray)
    return loaded


class TestRuntimeArtifactBindingUsesArtifactToLoad:
    """The per-run view feeds custom_loader/load the same way add_artifact_name() does."""

    def test_loads_runtime_value_when_artifact_key_is_not_alphabetically_smallest(self) -> None:
        shared_options = Options({})
        features = FeatureSet()
        features.add(Feature("a_col", shared_options))
        features.add(Feature("z_col", shared_options))

        view = _run_view(features, {"z_col": "runtime_value"})

        assert view.artifact_to_load == "z_col"
        assert view.name_of_one_feature == "a_col"
        assert BaseArtifact.custom_loader(view) == "runtime_value"

    def test_load_uses_artifact_to_load_when_bound_at_runtime(self) -> None:
        features = FeatureSet()
        features.add(Feature("a_col", Options({})))
        features.add(Feature("z_col", Options({})))

        assert BaseArtifact.load(_run_view(features, {"z_col": "runtime_value"})) == "runtime_value"

    def test_binding_is_name_and_value_or_none(self) -> None:
        features = FeatureSet()
        features.add(Feature("a_col", Options({})))
        features.add(Feature("z_col", Options({})))

        assert features.runtime_artifact_binding({"z_col": "v"}) == ArtifactBinding("z_col", "v")
        assert features.runtime_artifact_binding({"other": "v"}) is None

    def test_for_run_without_binding_keeps_prepare_time_mode(self) -> None:
        features = FeatureSet()
        features.add(Feature("z_col", Options({"z_col": "stored_value"})))
        features.add_artifact_name()

        view = features.for_run(None)

        assert view.artifact_to_load == "z_col"
        assert BaseArtifact.load(view) == "stored_value"

    def test_for_run_binding_switches_save_mode_to_load_on_the_view_only(self) -> None:
        features = FeatureSet()
        features.add(Feature("z_col", Options({})))
        features.add_artifact_name()
        assert features.artifact_to_save == "z_col"

        view = features.for_run(ArtifactBinding("z_col", "v"))

        assert (view.artifact_to_load, view.artifact_to_save, view.artifact_value) == ("z_col", None, "v")
        assert (features.artifact_to_load, features.artifact_to_save, features.artifact_value) == (None, "z_col", None)

    def test_for_run_view_write_does_not_reach_the_plan_set(self) -> None:
        features = FeatureSet()
        features.add(Feature("z_col", Options({})))

        view = features.for_run(None)
        view.save_artifact = "produced"

        assert features.save_artifact is None


class TestRuntimeArtifactBindingLeavesOptionsUntouched:
    """Binding a runtime artifact must not change Options or the hash of any Feature."""

    def test_feature_remains_a_member_after_binding(self) -> None:
        feature = Feature("abc")
        features = FeatureSet()
        features.add(feature)

        _run_view(features, {"abc": "ART"})

        assert feature in features.features

    def test_feature_options_and_hash_are_unchanged_after_binding(self) -> None:
        feature = Feature("abc")
        features = FeatureSet()
        features.add(feature)
        hash_before = hash(feature)

        _run_view(features, {"abc": "ART"})

        assert "abc" not in feature.options.group
        assert "abc" not in feature.options.context
        assert hash(feature) == hash_before

    def test_shared_options_are_unchanged_across_features(self) -> None:
        shared_options = Options({})
        features = FeatureSet()
        a_feature = Feature("a_col", shared_options)
        z_feature = Feature("z_col", shared_options)
        features.add(a_feature)
        features.add(z_feature)

        _run_view(features, {"z_col": "runtime_value"})

        assert "z_col" not in shared_options.group
        assert "z_col" not in shared_options.context
        assert a_feature in features.features
        assert z_feature in features.features

    def test_artifact_is_not_written_to_feature_options(self) -> None:
        shared_options = Options({})
        features = FeatureSet()
        features.add(Feature("a_col", shared_options))
        features.add(Feature("z_col", shared_options))

        _run_view(features, {"z_col": "runtime_value"})

        assert "z_col" not in shared_options.context


class TestRuntimeArtifactBindingRejectsGroupCollision:
    """A stored artifact key baked into Options.group (the "load via Options" pattern) must not
    silently coexist with a runtime artifact for the same name."""

    def test_raises_when_second_feature_has_group_baked_artifact(self) -> None:
        """The colliding key sits on the second feature's distinct Options, so every instance is checked."""
        clean_options = Options({})
        colliding_options = Options({"z_col": "stored_value"})
        features = FeatureSet()
        features.add(Feature("a_col", clean_options))
        features.add(Feature("z_col", colliding_options))

        with pytest.raises(ValueError, match="already stored in Options.group"):
            features.runtime_artifact_binding({"z_col": "new_runtime_value"})

        assert clean_options.context == {}
        assert "z_col" not in colliding_options.context
        assert features.artifact_to_load is None
        assert features.artifact_to_save is None


class TestRuntimeArtifactBindingAcrossDistinctOptionsInstances:
    """Value-equal but distinct Options instances: the binding is independent of them."""

    def test_runtime_value_loads_and_options_stay_untouched(self) -> None:
        options_a = Options({})
        options_z = Options({})
        fs = FeatureSet()
        fs.add(Feature("a_col", options_a))
        fs.add(Feature("z_col", options_z))

        view = _run_view(fs, {"z_col": "runtime_value"})

        assert options_a.context == {}
        assert options_z.context == {}
        assert BaseArtifact.load(view) == "runtime_value"

    def test_runtime_numpy_array_binds_and_re_binds(self) -> None:
        """numpy arrays are not boolean-comparable; a second binding with a different array succeeds."""
        fs = FeatureSet()
        fs.add(Feature("a_col", Options({})))
        fs.add(Feature("z_col", Options({})))

        first_array = np.array([1, 2, 3])
        first = _run_view(fs, {"z_col": first_array})
        assert np.array_equal(_loaded_array(first), first_array)

        second_array = np.array([4, 5, 6])
        second = _run_view(fs, {"z_col": second_array})
        assert np.array_equal(_loaded_array(second), second_array)
        assert np.array_equal(_loaded_array(first), first_array)
