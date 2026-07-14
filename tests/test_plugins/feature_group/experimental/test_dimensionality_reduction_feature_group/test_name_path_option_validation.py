"""Options-only algorithm parameters are validated on the string-named path too (issue #732).

``pca_svd_solver`` and ``tsne_method`` are strict_validation keys with an ``allowed_values``
space, and neither is ever encoded in the feature name. The config-based path rejected a bogus
value for them; the string-named path accepted it, because a PREFIX_PATTERN match short-circuited
before the options were ever read. Both paths must reach the same verdict.
"""

from __future__ import annotations

from mloda.provider import DefaultOptionKeys
from mloda.user import Options
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import (
    DimensionalityReductionFeatureGroup,
)


class TestPcaSvdSolverValidatedOnBothPaths:
    """The reproduction from issue #732."""

    def test_config_path_rejects_bogus_solver(self) -> None:
        """Config-based path already rejects a solver outside allowed_values."""
        options = Options(
            context={
                DimensionalityReductionFeatureGroup.ALGORITHM: "pca",
                DimensionalityReductionFeatureGroup.DIMENSION: 2,
                DefaultOptionKeys.in_features: "f0",
                DimensionalityReductionFeatureGroup.PCA_SVD_SOLVER: "bogus",
            }
        )

        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("placeholder", options) is False

    def test_name_path_rejects_bogus_solver(self) -> None:
        """String-named path must reject the same solver value, instead of waving it through."""
        options = Options(context={DimensionalityReductionFeatureGroup.PCA_SVD_SOLVER: "bogus"})

        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("f0,f1,f2__pca_2d", options) is False

    def test_name_path_accepts_valid_solver(self) -> None:
        """A member of the solver value space on a name-matched feature still matches."""
        options = Options(context={DimensionalityReductionFeatureGroup.PCA_SVD_SOLVER: "arpack"})

        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("f0__pca_2d", options) is True

    def test_name_path_rejection_reason_names_key_and_value(self) -> None:
        """The discarded ValueError message identifies the key and the rejected value."""
        options = Options(context={DimensionalityReductionFeatureGroup.PCA_SVD_SOLVER: "bogus"})

        reason = DimensionalityReductionFeatureGroup._strict_validation_rejection_reason("f0__pca_2d", options)

        assert reason is not None
        assert DimensionalityReductionFeatureGroup.PCA_SVD_SOLVER in reason
        assert "bogus" in reason


class TestTsneMethodValidatedOnBothPaths:
    """tsne_method has the same shape: strict, allowed_values, options-only."""

    def test_config_path_rejects_bogus_method(self) -> None:
        """Config-based path rejects a t-SNE method outside allowed_values."""
        options = Options(
            context={
                DimensionalityReductionFeatureGroup.ALGORITHM: "tsne",
                DimensionalityReductionFeatureGroup.DIMENSION: 2,
                DefaultOptionKeys.in_features: "f0",
                DimensionalityReductionFeatureGroup.TSNE_METHOD: "bogus",
            }
        )

        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("placeholder", options) is False

    def test_name_path_rejects_bogus_method(self) -> None:
        """String-named path must reject the same t-SNE method value."""
        options = Options(context={DimensionalityReductionFeatureGroup.TSNE_METHOD: "bogus"})

        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("f0__tsne_2d", options) is False

    def test_name_path_accepts_valid_method(self) -> None:
        """A member of the t-SNE method value space still matches."""
        options = Options(context={DimensionalityReductionFeatureGroup.TSNE_METHOD: "exact"})

        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("f0__tsne_2d", options) is True


class TestNamePathElementValidatorAndPresence:
    """dimension is strict + element_validator; presence stays satisfiable by the name."""

    def test_name_path_rejects_invalid_dimension_option(self) -> None:
        """An element_validator-backed key is validated on the name path as well."""
        options = Options(context={DimensionalityReductionFeatureGroup.DIMENSION: -1})

        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("f0__pca_2d", options) is False

    def test_name_match_without_options_still_matches(self) -> None:
        """Required-PRESENCE stays off on the name path: the name carries algorithm and dimension."""
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("f0__pca_2d", Options()) is True

    def test_non_strict_key_with_bad_looking_value_is_accepted(self) -> None:
        """isomap_n_neighbors is strict_validation=False, so no value space is enforced."""
        options = Options(context={DimensionalityReductionFeatureGroup.ISOMAP_N_NEIGHBORS: "not_a_number"})

        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("f0__isomap_2d", options) is True
