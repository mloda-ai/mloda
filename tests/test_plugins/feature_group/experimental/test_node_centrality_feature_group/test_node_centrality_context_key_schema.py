"""
Tests for NodeCentralityFeatureGroup.context_key_schema() opt-in.

These tests FAIL until the Green Agent adds:

    @classmethod
    def context_key_schema(cls):
        return cls.derive_context_key_schema()

to NodeCentralityFeatureGroup.
"""

import pytest

from mloda.provider import OptionsValidator
from mloda_plugins.feature_group.experimental.node_centrality.base import (
    NodeCentralityFeatureGroup,
)


class TestNodeCentralityContextKeySchema:
    def test_context_key_schema_is_opted_in(self) -> None:
        """context_key_schema() must return the PROPERTY_MAPPING-derived schema."""
        schema = NodeCentralityFeatureGroup.context_key_schema()
        assert schema is not None  # fails: base class returns None until opt-in
        assert schema == NodeCentralityFeatureGroup.derive_context_key_schema()
        assert NodeCentralityFeatureGroup.CENTRALITY_TYPE in schema
        assert NodeCentralityFeatureGroup.GRAPH_TYPE in schema
        assert NodeCentralityFeatureGroup.WEIGHT_COLUMN in schema

    def test_graph_typ_typo_is_caught(self) -> None:
        """A near-miss typo for graph_type must raise ValueError with suggestion."""
        schema = NodeCentralityFeatureGroup.context_key_schema()
        assert schema is not None  # fails: base class returns None until opt-in
        with pytest.raises(ValueError) as exc:
            OptionsValidator.validate_context_keys(
                "some_feature",
                {"graph_typ": "directed"},
                schema,
            )
        assert "graph_typ" in str(exc.value)
        assert "did you mean" in str(exc.value)
        assert "graph_type" in str(exc.value)
