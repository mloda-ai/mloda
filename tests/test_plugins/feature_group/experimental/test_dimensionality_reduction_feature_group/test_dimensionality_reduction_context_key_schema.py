"""
Tests for DimensionalityReductionFeatureGroup.context_key_schema() opt-in.

These tests FAIL until the Green Agent adds:

    @classmethod
    def context_key_schema(cls):
        return cls.derive_context_key_schema()

to DimensionalityReductionFeatureGroup.
"""

import pytest

from mloda.provider import OptionsValidator
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import (
    DimensionalityReductionFeatureGroup,
)


class TestDimensionalityReductionContextKeySchema:
    def test_context_key_schema_is_opted_in(self) -> None:
        """context_key_schema() must return the PROPERTY_MAPPING-derived schema."""
        schema = DimensionalityReductionFeatureGroup.context_key_schema()
        assert schema is not None  # fails: base class returns None until opt-in
        assert schema == DimensionalityReductionFeatureGroup.derive_context_key_schema()
        assert DimensionalityReductionFeatureGroup.ALGORITHM in schema
        assert DimensionalityReductionFeatureGroup.DIMENSION in schema
        assert DimensionalityReductionFeatureGroup.ISOMAP_N_NEIGHBORS in schema

    def test_isomap_n_neighbor_typo_is_caught(self) -> None:
        """A near-miss typo for isomap_n_neighbors must raise ValueError with suggestion."""
        schema = DimensionalityReductionFeatureGroup.context_key_schema()
        assert schema is not None  # fails: base class returns None until opt-in
        with pytest.raises(ValueError) as exc:
            OptionsValidator.validate_context_keys(
                "some_feature",
                {"isomap_n_neighbor": 5},
                schema,
            )
        assert "isomap_n_neighbor" in str(exc.value)
        assert "did you mean" in str(exc.value)
        assert "isomap_n_neighbors" in str(exc.value)
