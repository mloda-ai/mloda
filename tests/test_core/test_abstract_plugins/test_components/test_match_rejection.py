"""The match-rejection channel lives in a neutral module (issue #727, cycles 1 and 3).

``mloda.core.abstract_plugins.components.match_rejection`` owns ``MATCH_REJECTION_REASONS`` and
``record_match_rejection`` so input-data reader code can record rejections without depending on the
feature-chainer module. The documented ``feature_chain_parser`` seam keeps re-exporting the SAME
function object, the engine harvests through the new home, and ``mloda.provider`` exports the
recording function for plugin authors.

Cycle 3: the window value is a frozen ``MatchRejection`` dataclass carrying the reason and a stage
hint (default ``"value_rejection"``), so a reader decline can arrive stamped ``"input_data"``. The
neutral module does not validate the stage string. ``MatchRejection`` is imported inside the test
functions on purpose: while it does not exist yet, only these tests fail, not the whole collection.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from typing import Any

import pytest

from mloda import provider
from mloda.core.abstract_plugins.components import match_hook, match_rejection
from mloda.core.abstract_plugins.components.feature_chainer import feature_chain_parser
from mloda.core.abstract_plugins.components.match_rejection import (
    MATCH_REJECTION_REASONS,
    record_match_rejection,
)
from mloda.provider import record_match_rejection as provider_record_match_rejection


@pytest.fixture
def recording_window() -> Iterator[dict[str, Any]]:
    """Open a fresh recording window and always close it again, even when the test body fails."""
    reasons: dict[str, Any] = {}
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
        """The shared probe both seams read uses the neutral module's contextvar object, so the harvest works."""
        assert getattr(match_hook, "MATCH_REJECTION_REASONS") is match_rejection.MATCH_REJECTION_REASONS


class TestRecording:
    """Recording is a no-op outside a window and first-reason-per-owner-wins inside one."""

    def test_recording_outside_a_window_is_a_no_op(self) -> None:
        """With the contextvar at its default None, recording raises nothing and stores nothing."""
        record_match_rejection("InertOwner727", "inert reason 727")

        assert MATCH_REJECTION_REASONS.get() is None

    def test_the_first_reason_per_owner_wins_and_owners_stay_separate(self, recording_window: dict[str, Any]) -> None:
        """A second reason for the same owner never overwrites the first; a second owner gets its own slot."""
        from mloda.core.abstract_plugins.components.match_rejection import MatchRejection

        record_match_rejection("FirstOwner727", "first reason 727")
        record_match_rejection("FirstOwner727", "second reason 727")

        assert recording_window == {"FirstOwner727": MatchRejection(reason="first reason 727")}

        record_match_rejection("SecondOwner727", "other reason 727")

        assert recording_window == {
            "FirstOwner727": MatchRejection(reason="first reason 727"),
            "SecondOwner727": MatchRejection(reason="other reason 727"),
        }


class TestStageHint:
    """The window value carries a stage hint next to the reason (cycle 3)."""

    def test_the_neutral_module_exposes_matchrejection(self) -> None:
        """MatchRejection lives in the neutral module; its stage defaults to value_rejection."""
        from mloda.core.abstract_plugins.components.match_rejection import MatchRejection

        rejection = MatchRejection(reason="probe reason 727")

        assert rejection.reason == "probe reason 727"
        assert rejection.stage == "value_rejection"

    def test_recording_without_a_stage_stores_the_value_rejection_default(
        self, recording_window: dict[str, Any]
    ) -> None:
        """No stage argument: the stored MatchRejection carries the default value_rejection stage."""
        from mloda.core.abstract_plugins.components.match_rejection import MatchRejection

        record_match_rejection("DefaultStageOwner727", "default stage reason 727")

        stored = recording_window["DefaultStageOwner727"]
        assert stored == MatchRejection(reason="default stage reason 727", stage="value_rejection")

    def test_recording_with_a_stage_stores_that_stage(self, recording_window: dict[str, Any]) -> None:
        """stage='input_data' is stored verbatim on the MatchRejection."""
        from mloda.core.abstract_plugins.components.match_rejection import MatchRejection

        record_match_rejection("InputDataOwner727", "input data reason 727", stage="input_data")

        stored = recording_window["InputDataOwner727"]
        assert stored == MatchRejection(reason="input data reason 727", stage="input_data")

    def test_the_first_recording_per_owner_wins_across_differing_stages(self, recording_window: dict[str, Any]) -> None:
        """A later recording never overwrites the first, not even with a different stage."""
        from mloda.core.abstract_plugins.components.match_rejection import MatchRejection

        record_match_rejection("StageRaceOwner727", "first staged reason 727", stage="input_data")
        record_match_rejection("StageRaceOwner727", "second staged reason 727")

        expected = MatchRejection(reason="first staged reason 727", stage="input_data")
        assert recording_window == {"StageRaceOwner727": expected}


class TestProviderExport:
    """mloda.provider exposes the recording function for plugin authors."""

    def test_provider_exports_the_same_function(self) -> None:
        """The provider export is the neutral module's function object."""
        assert provider_record_match_rejection is match_rejection.record_match_rejection

    def test_provider_all_lists_the_export(self) -> None:
        """The export is part of the provider's public surface."""
        assert "record_match_rejection" in provider.__all__
