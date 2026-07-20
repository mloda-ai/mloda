"""Options-only algorithm parameters are validated on both match paths.

``pca_svd_solver`` and ``tsne_method`` are strict keys with an ``allowed_values`` space that the feature name
never encodes, so a name match must reach the same verdict as the config-based one.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise
from mloda.provider import DefaultOptionKeys
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import (
    DimensionalityReductionFeatureGroup,
)
from mloda_plugins.feature_group.experimental.dimensionality_reduction.pandas import (
    PandasDimensionalityReductionFeatureGroup,
)


class TestPcaSvdSolverValidatedOnBothPaths:
    """pca_svd_solver: strict, allowed_values, options-only."""

    def test_config_path_rejects_bogus_solver(self) -> None:
        """The config-based path rejects a solver outside allowed_values."""
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
        """The string-named path rejects the same solver value."""
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
        """The string-named path rejects the same t-SNE method value."""
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

    def test_isomap_n_neighbors_bad_value_is_rejected_on_the_name_path(self) -> None:
        """isomap_n_neighbors is strict + element_validator, so its value IS enforced (issue #724).

        This key used to declare an element_validator WITHOUT strict_validation, which never ran:
        dead config, so a bad value was accepted. ``PropertySpec`` now rejects that combination at
        construction and the key was made strict, which is what #724 pins. The guard against
        OVER-rejecting a genuinely unconstrained key lives in the core suite, on a purpose-built
        non-strict key (``test_name_path_validates_option_values.py::test_non_strict_key_accepts_any_value``).
        """
        options = Options(context={DimensionalityReductionFeatureGroup.ISOMAP_N_NEIGHBORS: "not_a_number"})

        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("f0__isomap_2d", options) is False


def _forwarded_child_options(algorithm: str) -> Options:
    """Child options as the engine builds them: the consumer's group option flows onto the child."""
    child = Options()
    child.inherit_from(Options(group={DimensionalityReductionFeatureGroup.ALGORITHM: algorithm}))
    return child


class TestForwardedAlgorithmOutsideTheChildValueSpace:
    """``algorithm`` is shared across feature group families with different value spaces. A forwarded value
    outside the child's own space is rejected as a value before the forwarded-name mismatch check sees it, so
    the verdict is a non-match. Pins which of the two messages wins.
    """

    def test_forwarded_out_of_space_value_is_a_non_match(self) -> None:
        """A non-match, not a raise: the value rejection precedes the mismatch check."""
        child_options = _forwarded_child_options("kmeans")
        assert child_options.inherited_group_keys == frozenset(
            {DimensionalityReductionFeatureGroup.ALGORITHM}
        )  # precondition

        result = DimensionalityReductionFeatureGroup.match_feature_group_criteria("f0__pca_2d", child_options)

        assert result is False

    def test_rejection_reason_names_key_and_value(self) -> None:
        """The strict-validation reason explains WHICH key and WHICH value were rejected."""
        reason = DimensionalityReductionFeatureGroup._strict_validation_rejection_reason(
            "f0__pca_2d", _forwarded_child_options("kmeans")
        )

        assert reason is not None
        assert DimensionalityReductionFeatureGroup.ALGORITHM in reason
        assert "kmeans" in reason

    def test_engine_error_names_the_rejected_key_and_value(self) -> None:
        """End-to-end the user still learns WHICH key and value lost them the match.

        The speculative extra-group-option hint is gone (#791/#782): it needed a second, speculative
        match pass, which the single-pass renderer does not do. The value-rejection line survives and
        carries the actionable detail.
        """
        feature = Feature("f0__pca_2d", _forwarded_child_options("kmeans"))
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            PandasDimensionalityReductionFeatureGroup: {PandasDataFrame},
        }

        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        message = str(exc_info.value)
        assert "No feature groups found" in message
        assert DimensionalityReductionFeatureGroup.ALGORITHM in message
        assert "kmeans" in message


class TestForwardedAlgorithmInsideTheChildValueSpace:
    """A forwarded value the child DOES accept still contradicts the name: that stays a hard error."""

    def test_forwarded_in_space_value_still_raises_the_mismatch_error(self) -> None:
        """'tsne' is a valid dimensionality reduction algorithm, but the name parses to 'pca'."""
        child_options = _forwarded_child_options("tsne")

        with pytest.raises(ValueError) as exc_info:
            DimensionalityReductionFeatureGroup.match_feature_group_criteria("f0__pca_2d", child_options)

        message = str(exc_info.value)
        assert "forward_group_exclude" in message
        assert DimensionalityReductionFeatureGroup.ALGORITHM in message
        assert "tsne" in message
        assert "pca" in message

    def test_engine_surfaces_actionable_guidance_for_the_in_space_mismatch(self) -> None:
        """However the engine reports it, the user is told to carve the key out."""
        feature = Feature("f0__pca_2d", _forwarded_child_options("tsne"))
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            PandasDimensionalityReductionFeatureGroup: {PandasDataFrame},
        }

        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        message = str(exc_info.value)
        assert "forward_group_exclude" in message
        assert DimensionalityReductionFeatureGroup.ALGORITHM in message
