"""Shared fixtures for the experimental feature group plugin tests."""

from collections.abc import Iterator

import pytest

from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS, MatchRejection


@pytest.fixture
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a per-test recording window and always close it again."""
    reasons: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(reasons)
    yield reasons
    MATCH_REJECTION_REASONS.reset(token)
