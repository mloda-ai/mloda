"""Allowlist contrast: string-parsed vs config-based ``FeatureChainParserMixin.input_features``.

Issue #579 establishes an allowlist default: nothing auto-forwards a consumer's group
options onto its source features unless explicitly opted in. Config-based in_features
(``options.get_in_features()``) are explicit user-provided features, so they follow the
allowlist default and do NOT auto-forward.

The one sanctioned exception is the string-parse branch of ``input_features``: a feature
name like ``source__op1_test`` is parsed into an in_feature string, and a plain string
cannot carry an explicit ``forward_group`` flag. To preserve same-family chain resolution,
that branch opts its parsed source features into forwarding (``forward_group=True``) by
default.

These tests pin both halves of that contrast so a future change cannot silently widen the
config-based fallback into auto-forwarding again.
"""

from __future__ import annotations

from uuid import uuid4

from mloda.provider import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.user import FeatureName
from mloda.user import Features
from mloda.user import Options


class MockFeatureGroup(FeatureChainParserMixin):
    """Mock Feature group mirroring the pattern used across the mixin test suite."""

    PREFIX_PATTERN = r".*__([\w]+)_test$"
    PROPERTY_MAPPING = {
        "operation": {
            "op1": "Operation 1",
            "op2": "Operation 2",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        }
    }


def test_string_parse_branch_forwards_group_options() -> None:
    """The string-parse branch opts its source feature into forwarding by default."""
    consumer_options = Options(group={"data_source": "prod", "operation": "op1"})
    feature_name = FeatureName("raw_col__op1_test")

    mock_fg = MockFeatureGroup()
    result = mock_fg.input_features(consumer_options, feature_name)
    assert result is not None
    assert len(result) == 1
    source_feature = next(iter(result))
    assert source_feature.forward_group is True

    Features([source_feature], child_options=consumer_options, child_uuid=uuid4())
    assert source_feature.options.group.get("data_source") == "prod"


def test_config_based_fallback_does_not_auto_forward() -> None:
    """The config-based fallback returns plain features that do not auto-forward."""
    consumer_options = Options(
        group={"data_source": "prod"},
        context={DefaultOptionKeys.in_features: "raw_col"},
    )
    # "unrelated_name" does not match PREFIX_PATTERN, so this forces the config fallback.
    feature_name = FeatureName("unrelated_name")

    mock_fg = MockFeatureGroup()
    result = mock_fg.input_features(consumer_options, feature_name)
    assert result is not None
    assert len(result) == 1
    for feature in result:
        assert not feature.forward_group, f"Expected {feature.name} to not auto-forward, got {feature.forward_group!r}"

    source_feature = next(iter(result))
    child_options = Options(group={"data_source": "prod", DefaultOptionKeys.in_features: "raw_col"})
    Features([source_feature], child_options=child_options, child_uuid=uuid4())

    assert "data_source" not in source_feature.options.group
