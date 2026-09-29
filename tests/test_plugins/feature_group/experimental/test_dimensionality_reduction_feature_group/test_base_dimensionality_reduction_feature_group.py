"""
Tests for the base DimensionalityReductionFeatureGroup class.
"""

from collections.abc import Iterator

import pytest

from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS, MatchRejection
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.provider import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.pandas import (
    PandasDimensionalityReductionFeatureGroup,
)


@pytest.fixture
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a per-test recording window and always close it again."""
    reasons: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(reasons)
    yield reasons
    MATCH_REJECTION_REASONS.reset(token)


class TestDimensionalityReductionFeatureGroup:
    """Tests for the DimensionalityReductionFeatureGroup class."""

    def test_extract_config_feature_with_double_underscore_name(self) -> None:
        options = Options(
            context={
                DimensionalityReductionFeatureGroup.ALGORITHM: "pca",
                DimensionalityReductionFeatureGroup.DIMENSION: 2,
                DefaultOptionKeys.in_features: frozenset([Feature("x1"), Feature("x2")]),
            }
        )
        algorithm, dimension, source_features, _ = (
            DimensionalityReductionFeatureGroup._extract_algorithm_dimension_and_source_features(
                Feature("a__b", options=options)
            )
        )
        assert algorithm == "pca"
        assert dimension == 2
        assert sorted(source_features) == ["x1", "x2"]

    def test_match_feature_group_criteria(self) -> None:
        """Test the match_feature_group_criteria method."""
        # Valid feature names
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("customer_metrics__pca_2d", Options())
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("product_features__tsne_3d", Options())
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("sensor_readings__isomap_5d", Options())

        # Invalid feature names
        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "customer_metrics__invalid_2d", Options()
        )
        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "customer_metrics__pca_invalid", Options()
        )
        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "customer_metrics_pca_2d", Options()
        )

    def test_chained_source_name_matches_and_extracts_the_chained_source(self) -> None:
        """The pattern parses from the last suffix, so a chained source is the whole prefix."""
        name = "s__mean_imputed__pca_2d"
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria(name, Options())
        assert DimensionalityReductionFeatureGroup._extract_source_features(Feature(name)) == ["s__mean_imputed"]

    def test_zero_dimension_is_a_recorded_non_match_naming_the_dimension(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria("x__pca_0d", Options())
        recorded = rejection_window["DimensionalityReductionFeatureGroup"]
        assert "dimension" in recorded.reason

    def test_valid_name_records_no_rejection(self, rejection_window: dict[str, MatchRejection]) -> None:
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("x__pca_2d", Options())
        assert rejection_window == {}

    @pytest.mark.parametrize("name", ["x__pca_2d", "s__mean_imputed__pca_2d"])
    def test_extract_dim_reduction_params_reads_the_named_captures(self, name: str) -> None:
        algorithm, dimension, _ = DimensionalityReductionFeatureGroup._extract_dim_reduction_params(Feature(name))
        assert algorithm == "pca"
        assert dimension == 2
        assert isinstance(dimension, int)

    def test_parse_reduction_suffix(self) -> None:
        """Test the parse_reduction_suffix method."""
        # Valid feature names
        algorithm, dimension = DimensionalityReductionFeatureGroup.parse_reduction_suffix("customer_metrics__pca_2d")
        assert algorithm == "pca"
        assert dimension == 2

        algorithm, dimension = DimensionalityReductionFeatureGroup.parse_reduction_suffix("product_features__tsne_3d")
        assert algorithm == "tsne"
        assert dimension == 3

        # Invalid feature names
        with pytest.raises(ValueError):
            DimensionalityReductionFeatureGroup.parse_reduction_suffix("customer_metrics__invalid_2d")

        with pytest.raises(ValueError):
            DimensionalityReductionFeatureGroup.parse_reduction_suffix("customer_metrics__pca_invalid")

        with pytest.raises(ValueError):
            DimensionalityReductionFeatureGroup.parse_reduction_suffix("customer_metrics_pca_2d")

    def test_umap_is_not_declared(self) -> None:
        """umap is not implemented by any compute framework, so it must not be declared."""
        assert "umap" not in DimensionalityReductionFeatureGroup.REDUCTION_ALGORITHMS

    def test_umap_feature_does_not_match(self) -> None:
        """A umap feature must fail at planning time, not at compute time."""
        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria("customer__umap_2d", Options())

        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "placeholder",
            Options(
                context={
                    DimensionalityReductionFeatureGroup.ALGORITHM: "umap",
                    DimensionalityReductionFeatureGroup.DIMENSION: 2,
                    DefaultOptionKeys.in_features: "customer_metrics",
                }
            ),
        )

    def test_declared_algorithms_are_dispatched_by_pandas(self) -> None:
        """Every declared algorithm must be computable by the pandas implementation."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("sklearn")

        data = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "f2": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0, 12.0, 11.0],
                "f3": [5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0, 4.0, 6.0, 12.0, 10.0, 11.0],
                "target": ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
            }
        )
        source_features = ["f1", "f2", "f3"]

        # _perform_reduction receives boundary-materialized options at runtime (#796); mirror that here.
        options = PandasDimensionalityReductionFeatureGroup.options_with_defaults(Options())
        for algorithm in DimensionalityReductionFeatureGroup.REDUCTION_ALGORITHMS:
            result = PandasDimensionalityReductionFeatureGroup._perform_reduction(
                data, algorithm, 2, source_features, options
            )
            assert result.shape == (12, 2), f"Unexpected result shape for algorithm {algorithm}"

    def test_input_features(self) -> None:
        """Test the input_features method."""
        feature_group = PandasDimensionalityReductionFeatureGroup()

        # Single source feature
        input_features = feature_group.input_features(Options(), FeatureName("customer_metrics__pca_2d"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("customer_metrics") in input_features

        # Multiple source features (comma-separated)
        input_features = feature_group.input_features(Options(), FeatureName("feature1,feature2,feature3__pca_2d"))
        assert input_features is not None
        assert len(input_features) == 3
        assert Feature("feature1") in input_features
        assert Feature("feature2") in input_features
        assert Feature("feature3") in input_features
