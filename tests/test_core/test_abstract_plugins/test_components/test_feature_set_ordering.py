"""Contract tests for deterministic, sorted-tuple name accessors on FeatureSet (#613 follow-up).

These pin the NEW API:
- get_all_names() -> tuple[str, ...] sorted alphabetically
- get_initial_requested_features() -> tuple[FeatureName, ...] sorted alphabetically
- get_sorted_features() -> tuple[Feature, ...] sorted by feature.name
- name_of_one_feature / get_name_of_one_feature() -> alphabetically smallest name, order-independent
"""

import itertools

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet


class TestGetAllNamesSortedTuple:
    def test_returns_alphabetically_sorted_names(self) -> None:
        features = FeatureSet()
        features.add(Feature("c_col"))
        features.add(Feature("a_col"))
        features.add(Feature("b_col"))

        assert features.get_all_names() == ("a_col", "b_col", "c_col")

    def test_returns_tuple(self) -> None:
        features = FeatureSet()
        features.add(Feature("c_col"))
        features.add(Feature("a_col"))
        features.add(Feature("b_col"))

        assert isinstance(features.get_all_names(), tuple)


class TestGetInitialRequestedFeaturesSortedTuple:
    def test_returns_alphabetically_sorted_requested_names_only(self) -> None:
        features = FeatureSet()
        features.add(Feature("z_req", initial_requested_data=True))
        features.add(Feature("m_req", initial_requested_data=True))
        features.add(Feature("not_req"))

        assert features.get_initial_requested_features() == (FeatureName("m_req"), FeatureName("z_req"))

    def test_returns_tuple(self) -> None:
        features = FeatureSet()
        features.add(Feature("z_req", initial_requested_data=True))
        features.add(Feature("m_req", initial_requested_data=True))
        features.add(Feature("not_req"))

        assert isinstance(features.get_initial_requested_features(), tuple)


class TestGetSortedFeatures:
    def test_returns_features_sorted_by_name(self) -> None:
        features = FeatureSet()
        features.add(Feature("c_col"))
        features.add(Feature("a_col"))
        features.add(Feature("b_col"))

        sorted_features = features.get_sorted_features()

        assert tuple(feature.name for feature in sorted_features) == ("a_col", "b_col", "c_col")

    def test_returns_tuple_of_features(self) -> None:
        features = FeatureSet()
        features.add(Feature("b_col"))
        features.add(Feature("a_col"))

        sorted_features = features.get_sorted_features()

        assert isinstance(sorted_features, tuple)
        assert all(isinstance(feature, Feature) for feature in sorted_features)


class TestNameOfOneFeatureDeterministic:
    def test_out_of_order_add_picks_alphabetically_smallest_attribute(self) -> None:
        features = FeatureSet()
        features.add(Feature("c_col"))
        features.add(Feature("a_col"))
        features.add(Feature("b_col"))

        assert features.name_of_one_feature == FeatureName("a_col")

    def test_out_of_order_add_picks_alphabetically_smallest_via_getter(self) -> None:
        features = FeatureSet()
        features.add(Feature("c_col"))
        features.add(Feature("a_col"))
        features.add(Feature("b_col"))

        assert features.get_name_of_one_feature() == FeatureName("a_col")

    def test_last_added_feature_is_not_always_the_result(self) -> None:
        # Guards against the old "last call wins" behavior: adding the alphabetically
        # smallest name first must not make it lose to a later, larger name.
        features = FeatureSet()
        features.add(Feature("a_col"))
        features.add(Feature("z_col"))

        assert features.get_name_of_one_feature() == FeatureName("a_col")

    def test_order_independent_across_explicit_orderings(self) -> None:
        names = ["c_col", "a_col", "b_col"]
        orderings = [names, list(reversed(names)), sorted(names), ["b_col", "c_col", "a_col"]]

        results = set()
        for ordering in orderings:
            features = FeatureSet()
            for name in ordering:
                features.add(Feature(name))
            results.add(features.get_name_of_one_feature())

        assert results == {FeatureName("a_col")}

    def test_order_independent_across_all_permutations(self) -> None:
        names = ["delta", "alpha", "charlie", "bravo"]
        expected = FeatureName(min(names))

        for ordering in itertools.permutations(names):
            features = FeatureSet()
            for name in ordering:
                features.add(Feature(name))

            assert features.get_name_of_one_feature() == expected
            assert features.name_of_one_feature == expected


class TestAddArtifactNameDeterministic:
    """add_artifact_name() must resolve artifact_to_save deterministically, regardless of add() order."""

    def test_artifact_to_save_is_alphabetically_smallest_name(self) -> None:
        features = FeatureSet()
        features.add(Feature("c_col"))
        features.add(Feature("a_col"))
        features.add(Feature("b_col"))

        features.add_artifact_name()

        assert features.artifact_to_save == "a_col"

    def test_artifact_to_save_matches_across_explicit_orderings(self) -> None:
        names = ["c_col", "a_col", "b_col"]
        orderings = [names, list(reversed(names)), sorted(names), ["b_col", "c_col", "a_col"]]

        results = set()
        for ordering in orderings:
            features = FeatureSet()
            for name in ordering:
                features.add(Feature(name))
            features.add_artifact_name()
            results.add(features.artifact_to_save)

        assert results == {"a_col"}

    def test_artifact_to_save_matches_across_all_permutations(self) -> None:
        names = ["delta", "alpha", "charlie", "bravo"]
        expected = min(names)

        for ordering in itertools.permutations(names):
            features = FeatureSet()
            for name in ordering:
                features.add(Feature(name))
            features.add_artifact_name()

            assert features.artifact_to_save == expected


class TestNameOfOneFeatureMatchesSortedNamesInvariant:
    def test_name_of_one_feature_equals_first_of_get_all_names(self) -> None:
        features = FeatureSet()
        features.add(Feature("z_col"))
        features.add(Feature("m_col"))
        features.add(Feature("d_col"))
        features.add(Feature("a_col"))

        assert features.name_of_one_feature == features.get_all_names()[0]


class TestFeatureSetConstructorConvergesToAlphabeticallySmallest:
    """FeatureSet(features=[...]) is a distinct entry point into add(); must converge like direct .add() calls."""

    def test_name_of_one_feature_via_constructor(self) -> None:
        features = FeatureSet(features=[Feature("c_col"), Feature("a_col"), Feature("b_col")])

        assert features.name_of_one_feature == FeatureName("a_col")

    def test_add_artifact_name_via_constructor(self) -> None:
        features = FeatureSet(features=[Feature("c_col"), Feature("a_col"), Feature("b_col")])

        features.add_artifact_name()

        assert features.artifact_to_save == "a_col"
