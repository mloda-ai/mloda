"""The match-rejection channel lives in a neutral module (issue #727, cycles 1 and 3).

``mloda.core.abstract_plugins.components.match_rejection`` owns ``MATCH_REJECTION_REASONS`` and
``record_match_rejection`` so input-data reader code can record rejections without depending on the
feature-chainer module. The documented ``feature_chain_parser`` seam keeps re-exporting the SAME
function object, the engine harvests through the new home, and ``mloda.provider`` exports the
recording function, the ``collect_match_rejections`` collector and ``MatchRejection``.

Cycle 3: the window value is a frozen ``MatchRejection`` dataclass carrying the reason and a stage
hint (default ``"value_rejection"``), so a reader decline can arrive stamped ``"input_data"``. The
neutral module does not validate the stage string.
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
    MatchRejection,
    drop_match_rejections_since,
    match_rejection_owners,
    record_match_rejection,
)
from mloda.provider import collect_match_rejections
from tests.docs_corpus import REPO_ROOT


@pytest.fixture
def recording_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a fresh recording window and always close it again, even when the test body fails."""
    with collect_match_rejections() as reasons:
        yield reasons


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

        rejection = MatchRejection(reason="probe reason 727")

        assert rejection.reason == "probe reason 727"
        assert rejection.stage == "value_rejection"

    def test_recording_without_a_stage_stores_the_value_rejection_default(
        self, recording_window: dict[str, Any]
    ) -> None:
        """No stage argument: the stored MatchRejection carries the default value_rejection stage."""

        record_match_rejection("DefaultStageOwner727", "default stage reason 727")

        stored = recording_window["DefaultStageOwner727"]
        assert stored == MatchRejection(reason="default stage reason 727", stage="value_rejection")

    def test_recording_with_a_stage_stores_that_stage(self, recording_window: dict[str, Any]) -> None:
        """stage='input_data' is stored verbatim on the MatchRejection."""

        record_match_rejection("InputDataOwner727", "input data reason 727", stage="input_data")

        stored = recording_window["InputDataOwner727"]
        assert stored == MatchRejection(reason="input data reason 727", stage="input_data")

    def test_the_first_recording_per_owner_wins_across_differing_stages(self, recording_window: dict[str, Any]) -> None:
        """A later recording never overwrites the first, not even with a different stage."""

        record_match_rejection("StageRaceOwner727", "first staged reason 727", stage="input_data")
        record_match_rejection("StageRaceOwner727", "second staged reason 727")

        expected = MatchRejection(reason="first staged reason 727", stage="input_data")
        assert recording_window == {"StageRaceOwner727": expected}


class TestRefusalField:
    """A rejection can carry the extender exception that caused it, outside equality and repr."""

    def test_refusal_defaults_to_none(self) -> None:

        assert MatchRejection(reason="r").refusal is None

    def test_equality_and_repr_ignore_the_refusal(self) -> None:

        with_refusal = MatchRejection(reason="r", refusal=PermissionError("secret refusal"))

        assert with_refusal == MatchRejection(reason="r")
        assert "secret refusal" not in repr(with_refusal)

    def test_recording_stores_the_refusal_object(self, recording_window: dict[str, Any]) -> None:
        refusal = PermissionError("no clearance")

        record_match_rejection("RefusalOwner", "refused", stage="input_data", refusal=refusal)

        assert recording_window["RefusalOwner"].refusal is refusal


class TestMatchRejectionOwnersAndDrop:
    """The owner snapshot and the delta drop scope what one probe recorded."""

    OWNER = "vg1006_unit_owner"
    OTHER_OWNER = "vg1006_unit_other_owner"
    REASON = "vg1006 unit reason"

    def test_owners_outside_a_window_are_empty_and_open_none(self) -> None:
        assert MATCH_REJECTION_REASONS.get() is None
        assert match_rejection_owners() == frozenset()
        assert MATCH_REJECTION_REASONS.get() is None

    def test_owners_inside_a_window_are_the_recorded_names(self, rejection_window: dict[str, MatchRejection]) -> None:
        record_match_rejection(self.OWNER, self.REASON, stage="input_data")
        record_match_rejection(self.OTHER_OWNER, self.REASON, stage="input_data")

        assert match_rejection_owners() == frozenset({self.OWNER, self.OTHER_OWNER})

    def test_drop_without_a_window_is_a_no_op(self) -> None:
        assert MATCH_REJECTION_REASONS.get() is None
        drop_match_rejections_since(frozenset())
        assert MATCH_REJECTION_REASONS.get() is None

    def test_drop_keeps_owners_in_the_snapshot(self, rejection_window: dict[str, MatchRejection]) -> None:
        record_match_rejection(self.OWNER, self.REASON, stage="input_data")
        drop_match_rejections_since(match_rejection_owners())

        assert set(rejection_window) == {self.OWNER}

    def test_drop_removes_owners_recorded_after_the_snapshot(self, rejection_window: dict[str, MatchRejection]) -> None:
        record_match_rejection(self.OWNER, self.REASON, stage="input_data")
        known_owners = match_rejection_owners()
        record_match_rejection(self.OTHER_OWNER, self.REASON, stage="input_data")
        drop_match_rejections_since(known_owners)

        assert set(rejection_window) == {self.OWNER}


class TestProviderExport:
    """mloda.provider exposes the recording function, the collector and the value type for plugin authors."""

    @pytest.mark.parametrize("name", ["record_match_rejection", "collect_match_rejections", "MatchRejection"])
    def test_provider_exports_the_same_object(self, name: str) -> None:
        """The provider export is the neutral module's object."""
        assert getattr(provider, name) is getattr(match_rejection, name)

    @pytest.mark.parametrize("name", ["record_match_rejection", "collect_match_rejections", "MatchRejection"])
    def test_provider_all_lists_the_export(self, name: str) -> None:
        """The export is part of the provider's public surface."""
        assert name in provider.__all__


class TestCollectMatchRejections:
    """collect_match_rejections opens a window, yields its dict and restores the previous value."""

    def test_records_inside_the_window_are_collected(self) -> None:
        with collect_match_rejections() as reasons:
            record_match_rejection("CollectOwner", "collect reason")

        assert reasons == {"CollectOwner": MatchRejection(reason="collect reason")}

    def test_exit_restores_the_previous_value(self) -> None:
        def body() -> dict[str, MatchRejection]:
            with collect_match_rejections() as reasons:
                record_match_rejection("BeforeExit", "before")
            assert MATCH_REJECTION_REASONS.get() is None
            record_match_rejection("AfterExit", "after")
            return reasons

        reasons = contextvars.Context().run(body)

        assert reasons == {"BeforeExit": MatchRejection(reason="before")}

    def test_nested_windows_stay_separate(self) -> None:
        with collect_match_rejections() as outer:
            record_match_rejection("OuterOne", "outer one")
            with collect_match_rejections() as inner:
                record_match_rejection("InnerOne", "inner one")
            assert MATCH_REJECTION_REASONS.get() is outer
            record_match_rejection("OuterTwo", "outer two")

        assert inner == {"InnerOne": MatchRejection(reason="inner one")}
        assert outer == {
            "OuterOne": MatchRejection(reason="outer one"),
            "OuterTwo": MatchRejection(reason="outer two"),
        }

    def test_exception_propagates_and_restores_the_previous_value(self) -> None:
        def body() -> None:
            with pytest.raises(RuntimeError, match="boom"):
                with collect_match_rejections():
                    raise RuntimeError("boom")
            assert MATCH_REJECTION_REASONS.get() is None

        contextvars.Context().run(body)

    def test_the_matching_guide_documents_the_helper(self) -> None:
        page = REPO_ROOT / "docs" / "docs" / "in_depth" / "feature-group-matching.md"

        assert "collect_match_rejections" in page.read_text(encoding="utf-8")
