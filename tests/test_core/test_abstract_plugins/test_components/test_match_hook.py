"""Issue #991: one helper owns the match-hook call, the marked-abort re-raise and the return coercion.

``call_match_hook`` answers with a ``MatchHookOutcome``: the coerced verdict, the raw hook return and the
contained raise. The three existing match-seam files stay the behavior net; this one pins the helper itself
and that both seams route through it. Probe classes are dropped before any assert, per tests/conftest.py.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import PropertyValueRejection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components import match_hook as match_hook_module
from mloda.core.abstract_plugins.components.match_hook import MatchHookOutcome, call_match_hook
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import escalate_match_abort
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.filter.filter_type_enum import FilterType
from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


# The probe's module binding of the helper: the one point where a spy can intercept the hook call.
HELPER_NAME = "call_match_hook"

PROBE_CLASS_NAME = "MatchHookProbeFG991"
PROBE_FEATURE = "match_hook_probe_feat_991"
HOST_FEATURE = "match_hook_host_feat_991"  # the resolved feature the filters are matched against
FILTER_FEATURE_A = "match_hook_filter_a_991"
FILTER_FEATURE_B = "match_hook_filter_b_991"

RAISE_TYPE_NAME = "RuntimeError"
RAISE_MESSAGE = "boom_991_match_hook_raised"
BOOL_RAISE_MESSAGE = "boom_991_bool_exploded"
ABORT_MESSAGE = "abort_991_marked_match_abort"
REJECTION_MESSAGE = "reject_991_property_value"

T = TypeVar("T")


class FalsyBool991:
    """A returned value that is not False and says no only through __bool__."""

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<FalsyBool991>"


class TruthyBool991:
    """The mirror: a returned value that is not True and says yes only through __bool__."""

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "<TruthyBool991>"


class ExplodingBool991:
    """A returned value whose truthiness test raises: reading it is itself a plugin call."""

    def __bool__(self) -> bool:
        raise RuntimeError(BOOL_RAISE_MESSAGE)

    def __repr__(self) -> str:
        # Fixed text: a snapshot must be able to show this value without triggering the raise.
        return "<ExplodingBool991>"


# Keyed by id so parametrize stays readable; each row is ONE object, so identity pins the raw return.
FALSY_NON_BOOL_RETURNS: dict[str, Any] = {
    "none": None,
    "zero": 0,
    "empty_string": "",
    "empty_list": [],
    "falsy_bool_object": FalsyBool991(),
}
TRUTHY_NON_BOOL_RETURNS: dict[str, Any] = {
    "non_empty_string": "yes",
    "one": 1,
    "non_empty_list": [1],
    "truthy_bool_object": TruthyBool991(),
}
FALSY_RETURNS: dict[str, Any] = {**FALSY_NON_BOOL_RETURNS, "literal_false": False}
TRUTHY_RETURNS: dict[str, Any] = {**TRUTHY_NON_BOOL_RETURNS, "literal_true": True}

# The hook returned nothing at all, so the outcome's raw return must be None.
_NOTHING = object()


def _shown(value: Any) -> str:
    """Type and repr of a value, so an identity assert can report it without a truthiness test."""
    return f"{type(value).__name__} {value!r}"


def _raise(exc: BaseException) -> Any:
    """Raise ``exc``; a named function, so a probe's answer holds no throwaway class."""
    raise exc


def _capture(call: Callable[[], T]) -> tuple[T | None, str | None]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (red-phase probe: an escape is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _make_probe_fg(answer: Callable[[], Any]) -> type[FeatureGroup]:
    """A throwaway group whose match hook returns, or raises, whatever ``answer`` does."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class MatchHookProbeFG991(FeatureGroup):
        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> Any:  # Any, not bool: a non-bool out of a `-> bool` hook is exactly what the helper coerces.
            return answer()

    return MatchHookProbeFG991


@dataclass(frozen=True)
class _OutcomeSnapshot:
    """Plain-data readout of one call_match_hook call. Holds no class and no exception object."""

    escaped: str | None
    escaped_is_the_raised_object: bool
    is_the_result_type: bool
    matched_is_true: bool
    matched_is_false: bool
    matched_shown: str
    returned_is_the_hook_value: bool
    returned_shown: str
    error_is_none: bool
    error_is_the_raised_object: bool
    error_type: str | None
    error_message: str | None


def _drive(
    answer: Callable[[], Any],
    returned: Any = _NOTHING,
    raised: BaseException | None = None,
) -> _OutcomeSnapshot:
    """Call the helper against one throwaway group and read the outcome out as plain data.

    ``returned`` is the object the hook hands back, ``raised`` the object it raises. The finally unbinds
    every name and clears every traceback that would pin the probe class.
    """
    fg = _make_probe_fg(answer)
    outcome: Any = None
    escaped: BaseException | None = None
    error: BaseException | None = None
    matched: Any = None
    raw: Any = None
    try:
        try:
            outcome = call_match_hook(fg, PROBE_FEATURE, Options(), None)
        except BaseException as exc:  # noqa: BLE001  (an escape, or its absence, is the fact under test)
            escaped = exc
        if outcome is not None:
            matched, raw, error = outcome.matched, outcome.returned, outcome.error
        expected = None if returned is _NOTHING else returned
        return _OutcomeSnapshot(
            escaped=None if escaped is None else f"{type(escaped).__name__}: {escaped}",
            escaped_is_the_raised_object=escaped is not None and escaped is raised,
            is_the_result_type=isinstance(outcome, MatchHookOutcome),
            matched_is_true=matched is True,
            matched_is_false=matched is False,
            matched_shown=_shown(matched),
            returned_is_the_hook_value=raw is expected,
            returned_shown=_shown(raw),
            error_is_none=outcome is not None and error is None,
            error_is_the_raised_object=error is not None and error is raised,
            error_type=None if error is None else type(error).__name__,
            error_message=None if error is None else str(error),
        )
    finally:
        for exception in (escaped, error, raised):
            if exception is not None:
                # The frames hold the helper's own `feature_group` local, which pins the probe class.
                exception.__traceback__ = None
        del fg, outcome, escaped, error, matched, raw
        gc.collect()


class TestTheHelperCoercesTheReturn:
    """One coercion for both seams: truthiness decides, and the verdict handed back is a real bool."""

    @pytest.mark.parametrize("returned_id", sorted(TRUTHY_RETURNS))
    def test_a_truthy_return_comes_back_as_true(self, returned_id: str) -> None:
        value = TRUTHY_RETURNS[returned_id]

        snapshot = _drive(lambda: value, returned=value)

        assert snapshot.escaped is None, f"nothing may cross the helper: {snapshot.escaped}"
        assert snapshot.is_the_result_type, f"the helper must answer with a MatchHookOutcome: {snapshot.matched_shown}"
        assert snapshot.matched_is_true, f"a truthy return must come back as True, not raw: {snapshot.matched_shown}"
        assert snapshot.error_is_none, f"a hook that returned contained nothing: {snapshot.error_type}"

    @pytest.mark.parametrize("returned_id", sorted(FALSY_RETURNS))
    def test_a_falsy_return_comes_back_as_false(self, returned_id: str) -> None:
        value = FALSY_RETURNS[returned_id]

        snapshot = _drive(lambda: value, returned=value)

        assert snapshot.escaped is None, f"nothing may cross the helper: {snapshot.escaped}"
        assert snapshot.is_the_result_type, f"the helper must answer with a MatchHookOutcome: {snapshot.matched_shown}"
        assert snapshot.matched_is_false, f"a falsy return must come back as False, not raw: {snapshot.matched_shown}"
        assert snapshot.error_is_none, f"a hook that returned contained nothing: {snapshot.error_type}"

    @pytest.mark.parametrize("returned_id", sorted(FALSY_NON_BOOL_RETURNS))
    def test_the_raw_return_survives_a_falsy_non_bool(self, returned_id: str) -> None:
        """The filter seam reports a falsy non-bool, so the value itself must reach it through the outcome."""
        value = FALSY_NON_BOOL_RETURNS[returned_id]

        snapshot = _drive(lambda: value, returned=value)

        assert snapshot.escaped is None, f"nothing may cross the helper: {snapshot.escaped}"
        assert snapshot.returned_is_the_hook_value, (
            f"the raw return must survive so the caller can report it, got: {snapshot.returned_shown}"
        )
        assert snapshot.matched_is_false, f"a falsy non-bool is still a non-match: {snapshot.matched_shown}"

    @pytest.mark.parametrize("returned_id", sorted(TRUTHY_NON_BOOL_RETURNS))
    def test_the_raw_return_survives_a_truthy_non_bool(self, returned_id: str) -> None:
        """Same in the other direction, so the raw return is not a falsy-only afterthought."""
        value = TRUTHY_NON_BOOL_RETURNS[returned_id]

        snapshot = _drive(lambda: value, returned=value)

        assert snapshot.escaped is None, f"nothing may cross the helper: {snapshot.escaped}"
        assert snapshot.returned_is_the_hook_value, f"the raw return must survive, got: {snapshot.returned_shown}"
        assert snapshot.matched_is_true, f"a truthy non-bool is still a match: {snapshot.matched_shown}"


class TestTheHelperContainsARaise:
    """The containment has one home: the raise comes back in the outcome, unless it is marked."""

    def test_a_plain_raise_is_contained_and_handed_back(self) -> None:
        """Each seam records it its own way, so the helper hands the exception over instead of judging it."""
        marker = RuntimeError(RAISE_MESSAGE)

        snapshot = _drive(partial(_raise, marker), raised=marker)

        assert snapshot.escaped is None, f"an unmarked raise must not cross the helper: {snapshot.escaped}"
        assert snapshot.matched_is_false, f"a raising hook is a non-match: {snapshot.matched_shown}"
        assert snapshot.error_is_the_raised_object, (
            f"the outcome must carry the raise itself, got: {snapshot.error_type}: {snapshot.error_message}"
        )
        assert snapshot.returned_is_the_hook_value, f"nothing came back from the hook: {snapshot.returned_shown}"

    def test_a_return_that_explodes_on_bool_is_contained_too(self) -> None:
        """bool() belongs inside the containment: reading a plugin's return is itself a plugin call (#927).

        The raw return is deliberately unpinned here: a caller reads it only when nothing raised.
        """
        snapshot = _drive(lambda: ExplodingBool991())

        assert snapshot.escaped is None, f"the raise must not cross the helper: {snapshot.escaped}"
        assert snapshot.matched_is_false, f"an unreadable return is a non-match: {snapshot.matched_shown}"
        assert snapshot.error_type == RAISE_TYPE_NAME, f"the coercion's raise must be contained: {snapshot.error_type}"
        assert snapshot.error_message == BOOL_RAISE_MESSAGE, (
            f"the outcome must carry the raise the coercion produced: {snapshot.error_message}"
        )

    def test_a_marked_abort_propagates_unchanged(self) -> None:
        """A framework-owned raise crosses the helper as the SAME object, so both seams keep escalating it."""
        marker = escalate_match_abort(RuntimeError(ABORT_MESSAGE))

        snapshot = _drive(partial(_raise, marker), raised=marker)

        assert snapshot.escaped == f"{RAISE_TYPE_NAME}: {ABORT_MESSAGE}", (
            f"a marked abort must not be contained, got: {snapshot.escaped}"
        )
        assert snapshot.escaped_is_the_raised_object, "the marked exception itself must escape, not a wrapper"


@dataclass(frozen=True)
class _ContainedErrorSnapshot:
    """Plain-data readout of the exception one contained raise hands back. Holds no exception object."""

    escaped: str | None
    is_the_raised_object: bool
    has_traceback: bool
    type_name: str | None
    message: str | None
    is_a_property_value_rejection: bool


def _drive_contained_error(marker: Exception) -> _ContainedErrorSnapshot:
    """Contain one raise and read the exception the outcome carries, traceback included, out as plain data."""
    fg = _make_probe_fg(partial(_raise, marker))
    outcome: MatchHookOutcome | None = None
    error: Exception | None = None
    try:
        outcome, escaped = _capture(partial(call_match_hook, fg, PROBE_FEATURE, Options(), None))
        error = None if outcome is None else outcome.error
        return _ContainedErrorSnapshot(
            escaped=escaped,
            is_the_raised_object=error is marker,
            has_traceback=error is not None and error.__traceback__ is not None,
            type_name=None if error is None else type(error).__name__,
            message=None if error is None else str(error),
            is_a_property_value_rejection=isinstance(error, PropertyValueRejection),
        )
    finally:
        # The frames hold the helper's own `feature_group` and `returned` locals, which pin the probe class.
        marker.__traceback__ = None
        del fg, outcome, error
        gc.collect()


class TestTheContainedRaiseCarriesNoTraceback:
    """The outcome crosses a module boundary, so what it carries must not keep the plugin's frames alive."""

    def test_the_contained_exception_arrives_without_its_traceback(self) -> None:
        """On the seams' own try, the exception died at the implicit del closing the except block."""
        marker = RuntimeError(RAISE_MESSAGE)

        snapshot = _drive_contained_error(marker)

        assert snapshot.escaped is None, f"an unmarked raise must not cross the helper: {snapshot.escaped}"
        assert snapshot.has_traceback is False, (
            "the traceback's frames hold the helper's own feature_group and returned locals, so an outcome the "
            "caller keeps pins the plugin class and the raw return; clear __traceback__ before handing it back"
        )
        assert snapshot.is_the_raised_object, "the exception itself must come back, never a copy or a wrapper"
        assert snapshot.type_name == RAISE_TYPE_NAME, f"the type must survive the strip: {snapshot.type_name}"
        assert snapshot.message == RAISE_MESSAGE, f"the message must survive the strip: {snapshot.message}"

    def test_a_subclass_raise_still_answers_isinstance(self) -> None:
        """The resolution seam tells a rejection from a defect by type, so a wrapper would silently reclassify it."""
        marker = PropertyValueRejection(REJECTION_MESSAGE)

        snapshot = _drive_contained_error(marker)

        assert snapshot.escaped is None, f"an unmarked raise must not cross the helper: {snapshot.escaped}"
        assert snapshot.has_traceback is False, "a rejection is handed back tracebackless like any contained raise"
        assert snapshot.is_a_property_value_rejection, (
            f"isinstance against the subclass must still hold after the strip, got: {snapshot.type_name}"
        )
        assert snapshot.message == REJECTION_MESSAGE, f"the message must survive the strip: {snapshot.message}"


def _asked(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, str]:
    """(group class name, feature name) of one helper call, read without assuming how it was spelled."""
    values = [*args, *kwargs.values()]
    group = next((value for value in values if isinstance(value, type) and issubclass(value, FeatureGroup)), None)
    name = next((value for value in values if isinstance(value, str)), "?")
    return ("?" if group is None else str(group.get_class_name()), str(name))


class _Spy:
    """Records one entry per helper call and delegates to the real helper, so the seam behaves normally."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> MatchHookOutcome:
        self.seen.append(_asked(args, kwargs))
        return call_match_hook(*args, **kwargs)


def test_the_filter_seam_calls_the_helper_once_per_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The filter seam routes through the probe, whose module binding is the interception point, once per filter."""
    assert hasattr(match_hook_module, HELPER_NAME), (
        f"{match_hook_module.__name__} must hold {HELPER_NAME}, the binding the probe routes the hook call "
        "through; nothing here can observe a call the probe makes itself."
    )
    spy = _Spy()
    monkeypatch.setattr(match_hook_module, HELPER_NAME, spy)
    fg = _make_probe_fg(lambda: True)
    global_filter = GlobalFilter()
    global_filter.add_filter(FILTER_FEATURE_A, FilterType.EQUAL, {"value": 1})
    global_filter.add_filter(FILTER_FEATURE_B, FilterType.EQUAL, {"value": 2})
    matched: Any = None
    escaped: str | None = None
    names: tuple[str, ...] = ()
    try:
        matched, escaped = _capture(partial(global_filter.identify_matched_filters, fg, Feature(HOST_FEATURE), None))
        names = () if matched is None else tuple(sorted(single.name for single in matched))
        seen = sorted(spy.seen)
    finally:
        del fg, global_filter, matched, spy
        gc.collect()

    assert escaped is None, f"nothing may cross identify_matched_filters: {escaped}"
    assert seen == [(PROBE_CLASS_NAME, FILTER_FEATURE_A), (PROBE_CLASS_NAME, FILTER_FEATURE_B)], (
        f"the filter seam must reach the hook only through {HELPER_NAME}, once per filter, got: {seen}"
    )
    assert names == (FILTER_FEATURE_A, FILTER_FEATURE_B), f"both filters must still attach, got: {list(names)}"


def test_the_resolution_seam_calls_the_helper_once_per_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolution seam routes through the probe, whose module binding is the interception point, per candidate."""
    assert hasattr(match_hook_module, HELPER_NAME), (
        f"{match_hook_module.__name__} must hold {HELPER_NAME}, the binding the probe routes the hook call "
        "through; nothing here can observe a call the probe makes itself."
    )
    spy = _Spy()
    monkeypatch.setattr(match_hook_module, HELPER_NAME, spy)
    fg = _make_probe_fg(lambda: True)
    identifier = IdentifyFeatureGroupClass()
    verdict: Any = None
    escaped: str | None = None
    try:
        verdict, escaped = _capture(
            partial(identifier._filter_feature_group_by_criteria, fg, Feature(PROBE_FEATURE), None)
        )
        seen = sorted(spy.seen)
    finally:
        del fg, identifier, spy
        gc.collect()

    assert escaped is None, f"nothing may cross the resolution seam: {escaped}"
    assert seen == [(PROBE_CLASS_NAME, PROBE_FEATURE)], (
        f"the resolution seam must reach the hook only through {HELPER_NAME}, once per candidate, got: {seen}"
    )
    assert verdict is True, f"the candidate still matches: {_shown(verdict)}"
