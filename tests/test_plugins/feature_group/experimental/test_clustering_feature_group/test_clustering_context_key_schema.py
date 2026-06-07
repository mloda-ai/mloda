"""
Tests for ClusteringFeatureGroup.context_key_schema() opt-in.

These tests FAIL until the Green Agent adds:

    @classmethod
    def context_key_schema(cls):
        return cls.derive_context_key_schema()

to ClusteringFeatureGroup.
"""

import pytest

from mloda.provider import OptionsValidator
from mloda_plugins.feature_group.experimental.clustering.base import (
    ClusteringFeatureGroup,
)


class TestClusteringContextKeySchema:
    def test_context_key_schema_is_opted_in(self) -> None:
        """context_key_schema() must return the PROPERTY_MAPPING-derived schema."""
        schema = ClusteringFeatureGroup.context_key_schema()
        assert schema is not None  # fails: base class returns None until opt-in
        assert schema == ClusteringFeatureGroup.derive_context_key_schema()
        assert ClusteringFeatureGroup.ALGORITHM in schema
        assert ClusteringFeatureGroup.K_VALUE in schema
        assert ClusteringFeatureGroup.OUTPUT_PROBABILITIES in schema

    def test_output_probabilites_typo_is_caught(self) -> None:
        """A near-miss typo for output_probabilities must raise ValueError with suggestion."""
        schema = ClusteringFeatureGroup.context_key_schema()
        assert schema is not None  # fails: base class returns None until opt-in
        with pytest.raises(ValueError) as exc:
            OptionsValidator.validate_context_keys(
                "some_feature",
                {"output_probabilites": True},
                schema,
            )
        assert "output_probabilites" in str(exc.value)
        assert "did you mean" in str(exc.value)
        assert "output_probabilities" in str(exc.value)
