"""Regression tests: resolve_feature exposes each candidate's decline code."""

from __future__ import annotations

from mloda.core.api.plugin_info import ResolvedFeature


def test_resolved_feature_decline_codes_defaults_empty() -> None:
    resolved = ResolvedFeature(
        feature_name="f",
        feature_group=None,
        candidates=[],
        error="no match",
    )
    assert resolved.decline_codes == {}


def test_resolved_feature_decline_codes_accepts_mapping() -> None:
    resolved = ResolvedFeature(
        feature_name="f",
        feature_group=None,
        candidates=[],
        error="no match",
        decline_codes={"SomeFG": "missing_option"},
    )
    assert resolved.decline_codes == {"SomeFG": "missing_option"}
