"""The match-rejection channel lives in a neutral module (issue #727, cycle 1).

``mloda.core.abstract_plugins.components.match_rejection`` owns ``MATCH_REJECTION_REASONS`` and
``record_match_rejection`` so input-data reader code can record rejections without depending on the
feature-chainer module. The documented ``feature_chain_parser`` seam keeps re-exporting the SAME
function object, the engine harvests through the new home, and ``mloda.provider`` exports the
recording function for plugin authors.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator

import pytest

from mloda import provider
from mloda.core.abstract_plugins.components import match_rejection
from mloda.core.abstract_plugins.components.feature_chainer import feature_chain_parser
from mloda.core.abstract_plugins.components.match_rejection import (
    MATCH_REJECTION_REASONS,
    record_match_rejection,
)
from mloda.core.prepare import identify_feature_group
from mloda.provider import record_match_rejection as provider_record_match_rejection


@pytest.fixture
def recording_window() -> Iterator[dict[str, str]]:
    """Open a fresh recording window and always close it again, even when the test body fails."""
    reasons: dict[str, str] = {}
    token = MATCH_REJECTION_REASONS.set(reasons)
    yield reasons
    MATCH_REJECTION_REASONS.reset(token)


class TestNeutralModuleOwnsTheChannel:
    """The neutral module defines the channel and every old home aliases its objects."""

    def test_the_neutral_module_exposes_the_channel(self) -> None:
        """The new module carries the contextvar and the recording function."""
        assert isinstance(match_rejection.MATCH_REJECTION_REASONS, contextvars.ContextVar)
        assert callable(match_rejection.record_match_rejection)

    def test_the_contextvar_defaults_to_inactive(self) -> None:
        """In a fresh context the recorder holds None: recording is off."""
        assert contextvars.Context().run(MATCH_REJECTION_REASONS.get) is None

    def test_feature_chain_parser_reexports_the_same_function(self) -> None:
        """The documented feature_chain_parser seam stays the SAME function object, not a copy."""
        assert getattr(feature_chain_parser, "record_match_rejection") is match_rejection.record_match_rejection

    def test_the_engine_harvests_through_the_new_home(self) -> None:
        """identify_feature_group uses the neutral module's contextvar object, so the harvest still works."""
        assert getattr(identify_feature_group, "MATCH_REJECTION_REASONS") is match_rejection.MATCH_REJECTION_REASONS


class TestRecording:
    """Recording is a no-op outside a window and first-reason-per-owner-wins inside one."""

    def test_recording_outside_a_window_is_a_no_op(self) -> None:
        """With the contextvar at its default None, recording raises nothing and stores nothing."""
        record_match_rejection("InertOwner727", "inert reason 727")

        assert MATCH_REJECTION_REASONS.get() is None

    def test_the_first_reason_per_owner_wins_and_owners_stay_separate(self, recording_window: dict[str, str]) -> None:
        """A second reason for the same owner never overwrites the first; a second owner gets its own slot."""
        record_match_rejection("FirstOwner727", "first reason 727")
        record_match_rejection("FirstOwner727", "second reason 727")

        assert recording_window == {"FirstOwner727": "first reason 727"}

        record_match_rejection("SecondOwner727", "other reason 727")

        assert recording_window == {
            "FirstOwner727": "first reason 727",
            "SecondOwner727": "other reason 727",
        }


class TestProviderExport:
    """mloda.provider exposes the recording function for plugin authors."""

    def test_provider_exports_the_same_function(self) -> None:
        """The provider export is the neutral module's function object."""
        assert provider_record_match_rejection is match_rejection.record_match_rejection

    def test_provider_all_lists_the_export(self) -> None:
        """The export is part of the provider's public surface."""
        assert "record_match_rejection" in provider.__all__
