"""Fixtures shared by the feature_chainer match tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS, MatchRejection
from mloda.core.abstract_plugins.components.options import Options


@pytest.fixture
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """An active rejection window, as the engine opens one around a candidate's match call."""
    reasons: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(reasons)
    yield reasons
    MATCH_REJECTION_REASONS.reset(token)


@pytest.fixture
def presence_checks(monkeypatch: pytest.MonkeyPatch) -> list[Options]:
    """Records the options each missing-required-keys check sees; clear it after defining classes."""
    seen: list[Options] = []
    original = FeatureChainParser._name_path_missing_required_keys

    def spy(cls: type[FeatureChainParser], effective_options: Options, property_mapping: dict[str, Any]) -> list[str]:
        seen.append(effective_options)
        return original(effective_options, property_mapping)

    monkeypatch.setattr(FeatureChainParser, "_name_path_missing_required_keys", classmethod(spy))
    return seen
