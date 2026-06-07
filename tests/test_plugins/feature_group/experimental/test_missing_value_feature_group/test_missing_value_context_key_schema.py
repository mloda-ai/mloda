"""
Tests for MissingValueFeatureGroup.context_key_schema() opt-in.

These tests FAIL until the Green Agent adds:

    @classmethod
    def context_key_schema(cls):
        return cls.derive_context_key_schema()

to MissingValueFeatureGroup.
"""

import pytest

from mloda.provider import OptionsValidator
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import (
    MissingValueFeatureGroup,
)


class TestMissingValueContextKeySchema:
    def test_context_key_schema_is_opted_in(self) -> None:
        """context_key_schema() must return the PROPERTY_MAPPING-derived schema."""
        schema = MissingValueFeatureGroup.context_key_schema()
        assert schema is not None  # fails: base class returns None until opt-in
        assert schema == MissingValueFeatureGroup.derive_context_key_schema()
        assert MissingValueFeatureGroup.IMPUTATION_METHOD in schema
        assert "constant_value" in schema
        assert "group_by_features" in schema

    def test_constant_valeu_typo_is_caught(self) -> None:
        """A near-miss typo for constant_value must raise ValueError with suggestion."""
        schema = MissingValueFeatureGroup.context_key_schema()
        assert schema is not None  # fails: base class returns None until opt-in
        with pytest.raises(ValueError) as exc:
            OptionsValidator.validate_context_keys(
                "some_feature",
                {"constant_valeu": 0},
                schema,
            )
        assert "constant_valeu" in str(exc.value)
        assert "did you mean" in str(exc.value)
        assert "constant_value" in str(exc.value)
