"""Contract tests for deterministic, sorted-tuple name accessors on FeatureSet (#613 follow-up).

These pin the NEW API:
- get_all_names() -> tuple[str, ...] sorted alphabetically
- get_initial_requested_features() -> tuple[FeatureName, ...] sorted alphabetically
- get_sorted_features() -> tuple[Feature, ...] sorted by feature.name
"""

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
