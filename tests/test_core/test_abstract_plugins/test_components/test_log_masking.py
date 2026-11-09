"""Record-level and install/chain/disable tests for run-scoped log record masking.

Inside a non-empty registered_secrets_scope every log record has the registered literals masked.
Process-global changes that cannot be undone in-process run in a subprocess.
"""

from __future__ import annotations

import contextvars
import json
import linecache
import logging
import re
import subprocess  # nosec B404
import sys
import threading
import traceback
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from contextvars import copy_context
from types import MappingProxyType
from typing import Any, cast
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components import log_masking
from mloda.core.abstract_plugins.components.credential import RegisteredCredential
from mloda.core.abstract_plugins.components.credential_scrub import (
    current_registered_escaped,
    current_registered_secrets,
    redact_exception,
    registered_secrets_scope,
    scrub_credentials,
)
from mloda.core.abstract_plugins.components.plan_freeze import ReadOnlyDict

_SECRET = "Zk8-secret-Vq3LmRt"  # nosec B105
_OTHER = "Pw5-other-Hy2NcXs"  # nosec B105
_STATE: tuple[tuple[str, Any], ...] = (
    ("_disabled", False),
    ("_warned", False),
    ("_installed_factory", None),
    ("_installed_make_record", None),
    ("_previous_factory", None),
    ("_previous_make_record", None),
    ("_process_wide", False),
    ("_union", frozenset()),
    ("_union_masks", (frozenset(), frozenset())),
    ("_fork_generation", 0),
)


def _plain_make_record(
    self: logging.Logger,
    name: str,
    level: int,
    fn: str,
    lno: int,
    msg: Any,
    args: Any,
    exc_info: Any,
    func: str | None = None,
    extra: Mapping[str, Any] | None = None,
    sinfo: str | None = None,
) -> logging.LogRecord:
    record = logging.getLogRecordFactory()(name, level, fn, lno, msg, args, exc_info, func, sinfo)
    if extra is not None:
        for key in extra:
            record.__dict__[key] = extra[key]
    return record


@pytest.fixture(autouse=True)
def _pristine_logging() -> Iterator[None]:
    saved_factory = logging.getLogRecordFactory()
    saved_make_record = logging.Logger.makeRecord
    saved_state = {name: getattr(log_masking, name) for name, _default in _STATE}
    saved_counts = log_masking._union_counts
    saved_lock = log_masking._lock
    log_masking._union_counts = {}
    saved_probe = dict(log_masking._probe_seen)
    logging.setLogRecordFactory(logging.LogRecord)
    setattr(logging.Logger, "makeRecord", _plain_make_record)
    for name, value in _STATE:
        setattr(log_masking, name, value)
    log_masking._probe_seen.update({"factory": False, "make_record": False})
    yield
    logging.setLogRecordFactory(saved_factory)
    setattr(logging.Logger, "makeRecord", saved_make_record)
    for name, value in saved_state.items():
        setattr(log_masking, name, value)
    log_masking._union_counts = saved_counts
    log_masking._lock = saved_lock
    log_masking._probe_seen.clear()
    log_masking._probe_seen.update(saved_probe)


class _Capture(logging.Handler):
    """Collects records without formatting them."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []
        # Not in logging's manager: pytest 9.2 attaches its handler to registered non-propagating loggers.
        self.logger = logging.Logger(f"lm_test.{uuid4().hex}.child")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.addHandler(self)

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def capture() -> Iterator[_Capture]:
    handler = _Capture()
    yield handler
    handler.logger.removeHandler(handler)


def _scope(*literals: str) -> registered_secrets_scope:
    return registered_secrets_scope(frozenset(literals))


class _Holder:
    def __str__(self) -> str:
        return f"holder({_SECRET})"


class _Boom:
    def __str__(self) -> str:
        raise ValueError("str() is broken")


class _ReprOnly:
    def __str__(self) -> str:
        return "safe"

    def __repr__(self) -> str:
        return f"repr({_SECRET})"


# Written out here, not imported: a form the code under test forgets must still fail the assertions.
_FORM_BUILDERS: dict[str, Callable[[str], str]] = {
    "raw": lambda literal: literal,
    "repr": lambda literal: repr(literal)[1:-1],
    "json_ascii": lambda literal: json.dumps(literal)[1:-1],
    "json_utf8": lambda literal: json.dumps(literal, ensure_ascii=False)[1:-1],
    "quote_forced_repr": lambda literal: repr(literal + '"')[1:-2],
}


def _forms(literal: str) -> list[str]:
    return [build(literal) for build in _FORM_BUILDERS.values()]


def _assert_no_literal(record: logging.LogRecord, literal: str) -> None:
    for form in _forms(literal):
        assert form not in record.getMessage()
        assert form not in str(record.msg)
        assert form not in repr(record.args)


_BACKSLASH = "ab\\cd'ef12345"
_QUOTES = "a\"b'c-12345"
_NEWLINE = "line1\nsecret-9"  # stands for a multi-line PEM key
_UNICODE = 'sec"re\\t-ä-1234'  # nosec B105  (raw, repr, ASCII JSON and UTF-8 JSON all differ)
_SINGLE_QUOTE = "pa'ssword-123"  # nosec B105  (inside a string with both quote kinds, repr escapes it as \')
_ECHO_LITERALS = {
    "backslash": _BACKSLASH,
    "quotes": _QUOTES,
    "newline": _NEWLINE,
    "unicode": _UNICODE,
    "single_quote": _SINGLE_QUOTE,
}
_NESTED_CASES = [
    pytest.param(literal, msg, args, id=f"{lit_id}_{shape_id}")
    for lit_id, literal in _ECHO_LITERALS.items()
    for shape_id, msg, args in (
        ("list", "cfg %s", ([literal],)),
        ("tuple", "cfg %s", ((literal,),)),
        ("list_and_int", "cfg %s and %s", ([literal], 1)),
        ("bytes", "b %s", (literal.encode(),)),
        ("json", "cfg %s", (json.dumps({"k": literal}),)),
        ("json_utf8", "cfg %s", (json.dumps({"k": literal}, ensure_ascii=False),)),
    )
]

_NO_LEAK_CASES = [
    *_NESTED_CASES,
    pytest.param(_SECRET, "login %s", (_SECRET,), id="str_arg"),
    pytest.param(_SECRET, f"login {_SECRET}", (), id="preformatted"),
    pytest.param(_SECRET, "login %r", (_SECRET,), id="repr"),
    pytest.param("abcdef123456", "pre abc%s", ("def123456",), id="template_arg_straddle"),
    pytest.param("abcdef123456", "%s%s", ("abc", "def123456"), id="two_arg_straddle"),
    pytest.param(_SECRET, "got %s", (_Holder(),), id="object_arg"),
    pytest.param(_SECRET, _Holder(), (), id="object_msg"),
    pytest.param("12", "n=%d", (12,), id="literal_vs_d_spec"),
    pytest.param("name", "%(name)s ok", ({"name": "x"},), id="literal_vs_format_key"),
    pytest.param("pa%ss-Q9", "p=%s", ("pa%ss-Q9",), id="percent_literal"),
]


@pytest.mark.parametrize("process_wide", [False, True], ids=["run_scoped", "process_wide"])
@pytest.mark.parametrize("literal, msg, args", _NO_LEAK_CASES)
def test_registered_literal_never_survives_in_a_record(
    capture: _Capture, literal: str, msg: Any, args: tuple[Any, ...], process_wide: bool
) -> None:
    if process_wide:
        log_masking.enable_process_wide_log_masking()
        with _HeldScope(literal):
            _outside(lambda: capture.logger.info(msg, *args))
    else:
        with _scope(literal):
            capture.logger.info(msg, *args)

    (record,) = capture.records
    _assert_no_literal(record, literal)


def test_template_is_kept_for_grouping_in_the_common_case(capture: _Capture) -> None:
    with _scope(_SECRET):
        capture.logger.info("login %s", _SECRET)

    (record,) = capture.records
    assert record.msg == "login %s"
    assert record.args == ("***",)
    assert record.getMessage() == "login ***"


def test_dict_arg_masks_values_and_keeps_keys(capture: _Capture) -> None:
    with _scope(_SECRET):
        capture.logger.info("u=%(user)s o=%(other)s", {"user": _SECRET, "other": "x"})

    (record,) = capture.records
    assert record.getMessage() == "u=*** o=x"
    assert isinstance(record.args, Mapping)
    assert set(record.args) == {"user", "other"}


def test_read_only_dict_arg_masks_like_a_plain_dict(capture: _Capture) -> None:

    with _scope(_SECRET):
        capture.logger.info("u=%(user)s o=%(other)s", ReadOnlyDict({"user": _SECRET, "other": "x"}, "options"))

    (record,) = capture.records
    assert record.getMessage() == "u=*** o=x"
    assert isinstance(record.args, Mapping)
    assert set(record.args) == {"user", "other"}


@pytest.mark.parametrize("wrap", [MappingProxyType, OrderedDict], ids=["mappingproxy", "ordereddict"])
def test_non_dict_mapping_arg_holding_a_literal_is_flattened(capture: _Capture, wrap: Any) -> None:
    with _scope(_SECRET):
        capture.logger.info("u=%(user)s o=%(other)s", wrap({"user": _SECRET, "other": "x"}))

    (record,) = capture.records
    assert record.getMessage() == "u=*** o=x"
    assert record.args == ()


def test_registered_credential_arg_renders_as_it_does_outside_a_run(capture: _Capture) -> None:
    credential = RegisteredCredential({"host": "db.example", "password": _SECRET})
    capture.logger.info("cred %s", credential)
    with _scope(_SECRET):
        capture.logger.info("cred %s", credential)

    outside, inside = capture.records
    assert inside.getMessage() == outside.getMessage()
    assert inside.args is credential


@pytest.mark.parametrize(
    "arg",
    [
        pytest.param(MappingProxyType({"a": "b"}), id="mappingproxy"),
        pytest.param(OrderedDict([("a", "b")]), id="ordereddict"),
    ],
)
def test_mapping_arg_holding_no_literal_is_left_as_the_same_object(capture: _Capture, arg: Any) -> None:
    expected = logging.LogRecord("n", logging.INFO, "", 0, "m %s", (arg,), None).getMessage()
    with _scope(_SECRET):
        capture.logger.info("m %s", arg)

    (record,) = capture.records
    assert record.args is arg
    assert record.getMessage() == expected


@pytest.mark.parametrize("msg", [{"a": "b"}, ["a"], ("a",)], ids=["dict", "list", "tuple"])
def test_container_msg_holding_no_literal_is_the_same_object(capture: _Capture, msg: Any) -> None:
    with _scope(_SECRET):
        capture.logger.info(msg)

    assert capture.records[0].msg is msg


class _Tagged(str):
    """A str subclass that carries an attribute plain str would drop."""


def test_str_subclass_msg_holding_no_literal_is_the_same_object(capture: _Capture) -> None:
    msg = _Tagged("plain text")
    with _scope(_SECRET):
        capture.logger.info(msg)

    assert capture.records[0].msg is msg


def test_empty_mapping_arg_is_harmless(capture: _Capture) -> None:
    with _scope(_SECRET):
        capture.logger.info("empty %s", {})

    assert capture.records[0].getMessage() == "empty {}"


def test_record_with_no_args_none_is_masked(capture: _Capture) -> None:
    with _scope(_SECRET):
        record = capture.logger.makeRecord(
            capture.logger.name, logging.INFO, "f", 1, f"x {_SECRET}", cast(Any, None), None
        )

    assert _SECRET not in record.getMessage()


@pytest.mark.parametrize("container", [{"k": _SECRET}, [_SECRET]], ids=["dict", "list"])
def test_container_msg_keeps_its_type(capture: _Capture, container: Any) -> None:
    with _scope(_SECRET):
        capture.logger.info(container)

    (record,) = capture.records
    assert type(record.msg) is type(container)
    assert _SECRET not in str(record.msg)
    assert _SECRET not in record.getMessage()


@pytest.mark.parametrize(
    "literal, msg, args",
    [
        pytest.param(_SECRET, "a %s %s", (_SECRET,), id="wrong_count"),
        pytest.param(_SECRET, "%(a)s %(b)s", ({"a": _SECRET},), id="missing_key"),
        pytest.param(_SECRET, f"x {_SECRET}", ("extra",), id="no_placeholder"),
        pytest.param(_SECRET, "x %s %s", (_Boom(), _SECRET), id="raising_str"),
        pytest.param(_BACKSLASH, [_BACKSLASH], ("stray",), id="container_msg_escaped_form"),
    ],
)
def test_malformed_call_is_masked_and_never_raises(
    capture: _Capture, literal: str, msg: Any, args: tuple[Any, ...]
) -> None:
    with _scope(literal):
        capture.logger.info(msg, *args)

    (record,) = capture.records
    for form in (literal, repr(literal)[1:-1]):
        assert form not in str(record.msg)
        assert form not in repr(record.args)


def test_malformed_call_with_repr_only_literal_does_not_keep_the_object(capture: _Capture) -> None:
    with _scope(_SECRET):
        capture.logger.warning("a %s %s", _ReprOnly())

    (record,) = capture.records
    for arg in cast(tuple[Any, ...], record.args):
        assert _SECRET not in str(arg)
        assert _SECRET not in repr(arg)
    assert _SECRET not in repr(record.args)


def test_exc_info_exception_is_masked(capture: _Capture) -> None:
    with _scope(_SECRET):
        try:
            raise ValueError(f"bad {_SECRET}")
        except ValueError:
            capture.logger.error("failed", exc_info=True)

    (record,) = capture.records
    assert record.exc_info is not None
    assert _SECRET not in str(record.exc_info[1])
    assert _SECRET not in logging.Formatter().format(record)


def test_exc_info_exception_group_message_is_masked(capture: _Capture) -> None:
    group = ExceptionGroup(f"bad {_SECRET}", [ValueError("m")])
    with _scope(_SECRET):
        capture.logger.error("x", exc_info=group)

    (record,) = capture.records
    assert record.exc_info is not None
    assert _SECRET not in cast(ExceptionGroup[Exception], record.exc_info[1]).message
    assert _SECRET not in logging.Formatter().format(record)


def test_logger_exception_outside_an_except_block_does_not_raise(capture: _Capture) -> None:
    with _scope(_SECRET):
        capture.logger.exception("no active exception %s", _SECRET)

    (record,) = capture.records
    assert _SECRET not in record.getMessage()


_STACK_LITERAL = "emit_with_stack_marker_zq9"


def emit_with_stack_marker_zq9(logger: logging.Logger) -> None:
    logger.info("with stack", stack_info=True)


def test_stack_info_is_masked(capture: _Capture) -> None:
    with _scope(_STACK_LITERAL):
        emit_with_stack_marker_zq9(capture.logger)

    (record,) = capture.records
    assert record.stack_info is not None
    assert _STACK_LITERAL not in record.stack_info


def _emit_stack_echoing(logger: logging.Logger, echoed: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Log with stack_info from a frame whose source line carries echoed, as a traceback line would."""
    filename = f"<stack-echo-{uuid4().hex}>"
    clean = "def emit(logger):\n    logger.info('with stack', stack_info=True)\n"
    shown = ["def emit(logger):\n", f"    logger.info('with stack', stack_info=True)  # {echoed}\n"]
    monkeypatch.setitem(linecache.cache, filename, (len(clean), None, shown, filename))
    namespace: dict[str, Any] = {}
    exec(compile(clean, filename, "exec"), namespace)  # nosec B102
    namespace["emit"](logger)


_ECHO_SINKS = ["redact_exception", "scrub_credentials", "msg", "arg", "extra", "exc_info", "stack_info"]
_ECHO_CASES = [
    pytest.param(literal, form, sink, id=f"{lit_id}-{form}-{sink}")
    for lit_id, literal in _ECHO_LITERALS.items()
    for form in _FORM_BUILDERS
    for sink in _ECHO_SINKS
    # a source line cannot hold a raw newline, so the injected stack text would stop at the break
    if not (sink == "stack_info" and "\n" in _FORM_BUILDERS[form](literal))
]


@pytest.mark.parametrize("literal, form, sink", _ECHO_CASES)
def test_every_echoed_form_of_a_registered_literal_is_masked_by_every_sink(
    capture: _Capture, monkeypatch: pytest.MonkeyPatch, literal: str, form: str, sink: str
) -> None:
    echoed = _FORM_BUILDERS[form](literal)
    forms = _forms(literal)
    with _scope(literal):
        if sink == "redact_exception":
            exc = ValueError(f"x {echoed} y")
            redact_exception(exc, frozenset({literal}))
            assert all(f not in str(exc) for f in forms)
            return
        if sink == "scrub_credentials":
            assert all(f not in scrub_credentials(f"x {echoed} y") for f in forms)
            return
        if sink == "msg":
            capture.logger.info(f"x {echoed} y")
        elif sink == "arg":
            capture.logger.info("x %s y", echoed)
        elif sink == "extra":
            capture.logger.info("m", extra={"tag": echoed})
        elif sink == "exc_info":
            try:
                raise ValueError(f"x {echoed} y")
            except ValueError:
                capture.logger.error("failed", exc_info=True)
        else:
            _emit_stack_echoing(capture.logger, echoed, monkeypatch)

    (record,) = capture.records
    _assert_no_literal(record, literal)
    if sink == "extra":
        assert all(f not in getattr(record, "tag") for f in forms)
    if sink == "exc_info":
        assert record.exc_info is not None
        rendered = logging.Formatter().format(record)
        assert all(f not in str(record.exc_info[1]) and f not in rendered for f in forms)
    if sink == "stack_info":
        assert record.stack_info is not None
        assert "# ***" in record.stack_info
        assert all(f not in record.stack_info for f in forms)


def test_str_extra_values_are_masked_and_other_values_untouched(capture: _Capture) -> None:
    holder = _Holder()
    with _scope(_SECRET):
        capture.logger.info("m", extra={"tag": _SECRET, "count": 5, "obj": holder})

    (record,) = capture.records
    assert _SECRET not in getattr(record, "tag")
    assert getattr(record, "count") == 5
    assert getattr(record, "obj") is holder


class _Sneaky(str):
    """A str subclass whose str() and format() render the registered literal."""

    def __str__(self) -> str:
        return _SECRET

    def __format__(self, spec: str) -> str:
        return _SECRET


def test_str_subclass_extra_value_is_normalised_to_plain_str(capture: _Capture) -> None:
    with _scope(_SECRET):
        capture.logger.info("m", extra={"sneaky": _Sneaky("x")})

    (record,) = capture.records
    assert type(getattr(record, "sneaky")) is str
    assert _SECRET not in logging.Formatter("%(message)s %(sneaky)s").format(record)


class _RaisingRepr(str):
    def __repr__(self) -> str:
        raise RuntimeError("repr is broken")


def test_literal_with_a_raising_repr_does_not_stay_registered() -> None:
    def run() -> None:
        with pytest.raises(RuntimeError):
            with _scope(_RaisingRepr(_SECRET)):
                pass
        assert current_registered_secrets() == frozenset()
        assert current_registered_escaped() == frozenset()

    contextvars.copy_context().run(run)


class _CallCounting:
    """Arg whose str and repr must never be called; counts any call."""

    def __init__(self) -> None:
        self.calls = 0

    def __str__(self) -> str:
        self.calls += 1
        return "counted"

    def __repr__(self) -> str:
        self.calls += 1
        return "counted"


@pytest.mark.parametrize("process_wide", [False, True], ids=["run_scoped", "process_wide_empty_union"])
@pytest.mark.parametrize("installed", [False, True], ids=["never_installed", "installed_before"])
def test_records_are_untouched_with_nothing_registered(capture: _Capture, installed: bool, process_wide: bool) -> None:
    if process_wide:
        log_masking.enable_process_wide_log_masking()
    if installed:
        with _scope(_SECRET):
            pass
    probe = _CallCounting()
    args = (_SECRET, probe)
    capture.logger.info("login %s %s", *args, extra={"tag": _SECRET})

    (record,) = capture.records
    assert record.args == args
    assert getattr(record, "tag") == _SECRET
    assert probe.calls == 0
    capture.logger.info(probe)
    assert capture.records[-1].msg is probe
    assert probe.calls == 0


class _CountingLiteral(str):
    """A str literal whose repr calls are counted."""

    repr_calls: list[int]

    def __new__(cls, value: str) -> _CountingLiteral:
        self = super().__new__(cls, value)
        self.repr_calls = []
        return self

    def __repr__(self) -> str:
        self.repr_calls.append(1)
        return str.__repr__(self)


@pytest.mark.parametrize("mode", ["run_scoped", "process_wide"])
def test_no_escaped_form_is_rebuilt_per_record(capture: _Capture, mode: str) -> None:
    literal = _CountingLiteral(_BACKSLASH)
    assert repr(literal)[1:-1] != literal
    literal.repr_calls.clear()

    class _LitHolder:
        def __str__(self) -> str:
            return f"holder({literal})"

    holder = _LitHolder()

    def log_many() -> None:
        for i in range(20):
            capture.logger.info("plain %s %s", i, "text")
        capture.logger.info("hold %s", holder)
        capture.logger.info("bytes %s", literal.encode())
        try:
            raise ValueError(f"bad {literal}")
        except ValueError:
            capture.logger.error("failed", exc_info=True)

    if mode == "run_scoped":
        with _scope(literal):
            baseline = len(literal.repr_calls)
            assert baseline > 0
            log_many()
            after = len(literal.repr_calls)
    else:
        log_masking.enable_process_wide_log_masking()
        with _HeldScope(literal):
            baseline = len(literal.repr_calls)
            assert baseline > 0
            _outside(log_many)
            after = len(literal.repr_calls)

    assert after == baseline
    assert len(capture.records) == 23
    for record in capture.records[-3:]:
        _assert_no_literal(record, literal)
    failed = capture.records[-1]
    assert failed.exc_info is not None
    assert all(form not in str(failed.exc_info[1]) for form in _forms(literal))


def test_empty_scope_does_not_install() -> None:
    with _scope():
        pass

    assert logging.getLogRecordFactory() is logging.LogRecord
    assert logging.Logger.makeRecord is _plain_make_record


def test_first_non_empty_scope_installs_and_ensure_is_idempotent() -> None:
    with _scope(_SECRET):
        factory = logging.getLogRecordFactory()
        make_record = logging.Logger.makeRecord
        assert factory is not logging.LogRecord
        assert make_record is not _plain_make_record
        log_masking.ensure_log_masking()
        log_masking.ensure_log_masking()

    assert logging.getLogRecordFactory() is factory
    assert logging.Logger.makeRecord is make_record


class _HeldScope:
    """A registered_secrets_scope held open on another thread until released."""

    def __init__(self, *literals: str) -> None:
        self._entered, self._release = threading.Event(), threading.Event()
        self._literals = literals
        self._thread = threading.Thread(target=self._hold)

    def _hold(self) -> None:
        with _scope(*self._literals):
            self._entered.set()
            self._release.wait(5)

    def __enter__(self) -> _HeldScope:
        self._thread.start()
        assert self._entered.wait(5)
        return self

    def release(self) -> None:
        self._release.set()
        self._thread.join(5)

    def __exit__(self, *exc: object) -> None:
        self.release()


def _outside(action: Callable[[], Any]) -> Any:
    """Run action on a fresh thread with no scope and return its result."""
    result: list[Any] = []
    thread = threading.Thread(target=lambda: result.append(action()))
    thread.start()
    thread.join(5)
    return result[0]


@pytest.mark.parametrize("enabled", [False, True], ids=["not_enabled", "enabled"])
def test_record_from_a_thread_outside_any_scope_is_untouched_unless_process_wide(
    capture: _Capture, enabled: bool
) -> None:
    if enabled:
        log_masking.enable_process_wide_log_masking()
    with _HeldScope(_SECRET):
        _outside(lambda: capture.logger.info("raw %s", _SECRET))

    (record,) = capture.records
    assert record.getMessage() == ("raw ***" if enabled else f"raw {_SECRET}")


def test_enable_is_idempotent_and_reported() -> None:
    assert log_masking.process_wide_log_masking_enabled() is False
    log_masking.enable_process_wide_log_masking()
    log_masking.enable_process_wide_log_masking()
    assert log_masking.process_wide_log_masking_enabled() is True


def test_enable_masks_extra_values_from_an_outside_thread(capture: _Capture) -> None:
    log_masking.enable_process_wide_log_masking()
    with _HeldScope(_SECRET):
        _outside(lambda: capture.logger.info("m", extra={"tag": _SECRET}))

    (record,) = capture.records
    assert getattr(record, "tag") == "***"


def test_enable_is_a_no_op_after_disable() -> None:
    log_masking.disable_log_masking_quietly()
    log_masking.enable_process_wide_log_masking()
    assert log_masking.process_wide_log_masking_enabled() is False


def test_a_run_context_also_masks_other_runs_literals_when_enabled(capture: _Capture) -> None:
    log_masking.enable_process_wide_log_masking()
    with _HeldScope(_OTHER):
        with _scope(_SECRET):
            capture.logger.info("a %s b %s", _SECRET, _OTHER)

    (record,) = capture.records
    assert record.getMessage() == "a *** b ***"


def test_a_run_context_uses_only_its_own_set_when_not_enabled(capture: _Capture) -> None:
    with _HeldScope(_OTHER):
        with _scope(_SECRET):
            capture.logger.info("a %s b %s", _SECRET, _OTHER)

    (record,) = capture.records
    assert record.getMessage() == f"a *** b {_OTHER}"


def test_one_scope_object_entered_twice_keeps_its_literal_until_the_last_exit() -> None:
    log_masking.enable_process_wide_log_masking()
    scope = _scope(_SECRET)
    scope.__enter__()
    scope.__enter__()
    scope.__exit__(None, None, None)
    assert _SECRET in log_masking._union
    scope.__exit__(None, None, None)
    assert log_masking._union == frozenset()
    assert log_masking._union_masks == (frozenset(), frozenset())
    assert log_masking._union_counts == {}


def test_a_scope_inherited_across_a_fork_does_not_release_the_child_counts() -> None:
    contextvars.copy_context().run(_inherited_scope_scenario)


def _inherited_scope_scenario() -> None:
    log_masking.enable_process_wide_log_masking()
    inherited = _scope(_SECRET)
    inherited.__enter__()
    log_masking._renew_lock_after_fork()
    assert log_masking._union == frozenset()
    assert log_masking._union_masks == (frozenset(), frozenset())
    child = _scope(_SECRET)
    child.__enter__()
    inherited.__exit__(None, None, None)
    assert _SECRET in log_masking._union
    child.__exit__(None, None, None)
    assert log_masking._union == frozenset()
    assert log_masking._union_masks == (frozenset(), frozenset())
    assert log_masking._union_counts == {}


def test_a_shared_literal_stays_in_the_union_until_the_last_scope_exits(capture: _Capture) -> None:
    log_masking.enable_process_wide_log_masking()
    first, second = _HeldScope(_SECRET), _HeldScope(_SECRET)
    with first, second:
        first.release()
        _outside(lambda: capture.logger.info("one %s", _SECRET))
        second.release()
        _outside(lambda: capture.logger.info("two %s", _SECRET))

    one, two = capture.records
    assert one.getMessage() == "one ***"
    assert two.getMessage() == f"two {_SECRET}"
    assert log_masking._union == frozenset()
    assert log_masking._union_counts == {}


def test_union_and_counts_are_empty_after_nested_scopes_exit() -> None:
    log_masking.enable_process_wide_log_masking()
    with _scope(_SECRET):
        with _scope(_SECRET, _OTHER):
            assert log_masking._union == frozenset({_SECRET, _OTHER})
    assert log_masking._union == frozenset()
    assert log_masking._union_masks == (frozenset(), frozenset())
    assert current_registered_escaped() == frozenset()
    assert log_masking._union_counts == {}


def test_a_scope_entered_before_enable_is_not_tracked_and_exits_cleanly(capture: _Capture) -> None:
    with _HeldScope(_SECRET) as held:
        log_masking.enable_process_wide_log_masking()
        _outside(lambda: capture.logger.info("raw %s", _SECRET))
        held.release()

    (record,) = capture.records
    assert record.getMessage() == f"raw {_SECRET}"
    assert log_masking._union == frozenset()
    assert log_masking._union_counts == {}


def test_the_union_does_not_leak_into_scrubbing_or_the_registered_set() -> None:
    log_masking.enable_process_wide_log_masking()
    with _HeldScope(_SECRET):
        assert _outside(current_registered_secrets) == frozenset()
        assert _outside(lambda: scrub_credentials(_SECRET)) == _SECRET


def test_a_union_masked_logger_exception_masks_the_exception_text(capture: _Capture) -> None:
    log_masking.enable_process_wide_log_masking()

    def log_failure() -> None:
        try:
            raise ValueError(f"bad {_SECRET}")
        except ValueError:
            capture.logger.exception("failed")

    with _HeldScope(_SECRET):
        _outside(log_failure)

    (record,) = capture.records
    assert record.exc_info is not None
    assert _SECRET not in str(record.exc_info[1])
    assert _SECRET not in logging.Formatter().format(record)


def _frames_below(marker: str) -> int | None:
    frames = traceback.extract_stack()
    indexes = [i for i, frame in enumerate(frames) if frame.name == marker]
    return len(frames) - indexes[-1] if indexes else None


def _log_once(logger: logging.Logger) -> None:
    logger.info("depth %s", _SECRET)


def _install_depth_probes(depths: dict[str, list[int]]) -> None:
    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        depth = _frames_below("_log_once")
        if depth is not None:
            depths["factory"].append(depth)
        return logging.LogRecord(*args, **kwargs)

    previous = logging.Logger.makeRecord

    def make_record(self: logging.Logger, *args: Any, **kwargs: Any) -> logging.LogRecord:
        depth = _frames_below("_log_once")
        if depth is not None:
            depths["make_record"].append(depth)
        return previous(self, *args, **kwargs)

    logging.setLogRecordFactory(factory)
    setattr(logging.Logger, "makeRecord", make_record)


def _assert_constant_depth(depths: dict[str, list[int]]) -> None:
    for key, seen in depths.items():
        assert seen, key
        assert len(set(seen)) == 1, (key, seen)


def test_chain_depth_is_constant_across_repeated_and_nested_scope_entries(capture: _Capture) -> None:
    depths: dict[str, list[int]] = {"factory": [], "make_record": []}
    _install_depth_probes(depths)

    for _ in range(3):
        with _scope(_SECRET):
            _log_once(capture.logger)
            with _scope(_OTHER):
                _log_once(capture.logger)

    _assert_constant_depth(depths)


def test_chain_depth_is_constant_across_concurrent_threads(capture: _Capture) -> None:
    depths: dict[str, list[int]] = {"factory": [], "make_record": []}
    _install_depth_probes(depths)
    barrier = threading.Barrier(4)

    def work() -> None:
        barrier.wait(5)
        for _ in range(3):
            with _scope(_SECRET):
                _log_once(capture.logger)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    _assert_constant_depth(depths)


def test_a_third_party_wrapper_around_ours_does_not_grow_the_chain(capture: _Capture) -> None:
    depths: dict[str, list[int]] = {"factory": [], "make_record": []}
    _install_depth_probes(depths)
    with _scope(_SECRET):
        pass
    ours_factory = logging.getLogRecordFactory()
    ours_make_record = logging.Logger.makeRecord

    def third_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        return ours_factory(*args, **kwargs)

    def third_make_record(self: logging.Logger, *args: Any, **kwargs: Any) -> logging.LogRecord:
        return ours_make_record(self, *args, **kwargs)

    logging.setLogRecordFactory(third_factory)
    setattr(logging.Logger, "makeRecord", third_make_record)
    depths["factory"].clear()
    depths["make_record"].clear()
    for _ in range(3):
        with _scope(_SECRET):
            _log_once(capture.logger)

    _assert_constant_depth(depths)
    assert logging.getLogRecordFactory() is third_factory
    assert logging.Logger.makeRecord is third_make_record
    assert _SECRET not in capture.records[-1].getMessage()


def test_a_factory_set_earlier_still_runs(capture: _Capture) -> None:
    def base(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = logging.LogRecord(*args, **kwargs)
        setattr(record, "base_ran", True)
        return record

    logging.setLogRecordFactory(base)
    with _scope(_SECRET):
        capture.logger.info("login %s", _SECRET)

    (record,) = capture.records
    assert getattr(record, "base_ran", False) is True
    assert record.getMessage() == "login ***"


def test_a_non_chaining_replacement_is_wrapped_again_at_the_next_outermost_entry(capture: _Capture) -> None:
    with _scope(_SECRET):
        pass
    logging.setLogRecordFactory(logging.LogRecord)
    setattr(logging.Logger, "makeRecord", _plain_make_record)

    with _scope(_SECRET):
        capture.logger.info("login %s", _SECRET, extra={"tag": _SECRET})

    (record,) = capture.records
    assert record.getMessage() == "login ***"
    assert _SECRET not in getattr(record, "tag")


def _install_picky_factory() -> None:
    def picky(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = logging.LogRecord(*args, **kwargs)
        if type(record.msg) is not str:
            raise TypeError("probe")
        return record

    logging.setLogRecordFactory(picky)


def _install_picky_make_record() -> None:
    def picky(
        self: logging.Logger, name: str, level: int, fn: str, lno: int, msg: Any, *rest: Any
    ) -> logging.LogRecord:
        if type(msg) is not str:
            raise TypeError("probe")
        return _plain_make_record(self, name, level, fn, lno, msg, *rest)

    setattr(logging.Logger, "makeRecord", picky)


@pytest.mark.parametrize(
    "install", [_install_picky_factory, _install_picky_make_record], ids=["factory", "make_record"]
)
def test_a_hook_that_raises_on_the_probe_never_breaks_scope_entry(
    capture: _Capture, install: Callable[[], None]
) -> None:
    install()

    def run() -> None:
        with _scope(_SECRET):
            capture.logger.info("login %s", _SECRET, extra={"tag": _SECRET})
        assert current_registered_secrets() == frozenset()

    copy_context().run(run)

    (record,) = capture.records
    assert record.getMessage() == "login ***"
    assert _SECRET not in getattr(record, "tag")


def test_repeated_entries_with_a_probe_raising_hook_do_not_grow_the_chain() -> None:
    _install_picky_factory()
    _install_picky_make_record()

    def enter_once() -> tuple[object, object]:
        with _scope(_SECRET):
            pass
        return logging.getLogRecordFactory(), logging.Logger.makeRecord

    first = copy_context().run(enter_once)
    for _ in range(3):
        assert copy_context().run(enter_once) == first


def test_wrappers_that_are_not_outermost_stay_in_place_but_inert_after_disable(capture: _Capture) -> None:
    with _scope(_SECRET):
        pass
    ours_factory = logging.getLogRecordFactory()
    ours_make_record = logging.Logger.makeRecord

    def third_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        return ours_factory(*args, **kwargs)

    def third_make_record(self: logging.Logger, *args: Any, **kwargs: Any) -> logging.LogRecord:
        return ours_make_record(self, *args, **kwargs)

    logging.setLogRecordFactory(third_factory)
    setattr(logging.Logger, "makeRecord", third_make_record)
    log_masking.disable_log_masking_quietly()

    with _scope(_SECRET):
        capture.logger.info("login %s", _SECRET, extra={"tag": _SECRET})

    (record,) = capture.records
    assert record.args == (_SECRET,)
    assert getattr(record, "tag") == _SECRET
    assert logging.getLogRecordFactory() is third_factory
    assert logging.Logger.makeRecord is third_make_record
    assert log_masking._installed_factory is ours_factory
    assert log_masking._installed_make_record is ours_make_record


def test_no_registered_literal_reaches_the_regex_cache(capture: _Capture) -> None:
    literal = f"lit-{uuid4().hex}"
    with _scope(literal):
        capture.logger.info("x %s", literal)
        capture.logger.info(f"pre {literal}")

    keys = list(getattr(re, "_cache", {})) + list(getattr(re, "_cache2", {}))
    assert not any(literal in repr(key) for key in keys)


def test_no_union_masked_literal_reaches_the_regex_cache(capture: _Capture) -> None:
    literal = f"lit-{uuid4().hex}"
    log_masking.enable_process_wide_log_masking()
    with _HeldScope(literal):
        _outside(lambda: capture.logger.info("x %s", literal))
        _outside(lambda: capture.logger.info(f"pre {literal}"))

    assert all(literal not in record.getMessage() for record in capture.records)
    keys = list(getattr(re, "_cache", {})) + list(getattr(re, "_cache2", {}))
    assert not any(literal in repr(key) for key in keys)


def _run_python(script: str, *flags: str) -> Any:
    done = subprocess.run(  # nosec B603
        [sys.executable, *flags, "-c", script], capture_output=True, text=True, timeout=8, check=False
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


_INHERIT_SCRIPT = f"""
import json, logging
from concurrent.futures import ThreadPoolExecutor
from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

secret = {_SECRET!r}
seen = []


class H(logging.Handler):
    def emit(self, record):
        seen.append(record.getMessage())


log = logging.getLogger("inherit")
log.setLevel(logging.DEBUG)
log.propagate = False
log.addHandler(H())
pool = ThreadPoolExecutor(max_workers=1)
with registered_secrets_scope(frozenset([secret])):
    pool.submit(lambda: None).result()
pool.submit(lambda: log.info("v %s", secret)).result()
print(json.dumps(seen))
"""


def test_pool_thread_with_a_stale_set_is_masked_against_it() -> None:
    if sys.version_info >= (3, 14):
        assert _run_python(_INHERIT_SCRIPT, "-X", "thread_inherit_context=1") == ["v ***"]
    else:
        assert _run_python(_INHERIT_SCRIPT) == [f"v {_SECRET}"]


_IMPORT_SCRIPT = """
import json, logging
factory = logging.getLogRecordFactory()
make_record = logging.Logger.makeRecord
import mloda, mloda.user, mloda.provider, mloda.steward
print(json.dumps([logging.getLogRecordFactory() is factory, logging.Logger.makeRecord is make_record]))
"""


def test_importing_mloda_changes_no_logging_global() -> None:
    assert _run_python(_IMPORT_SCRIPT) == [True, True]


_DISABLE_SCRIPT = f"""
import json, logging
from mloda.steward import disable_log_masking, enable_process_wide_log_masking
from mloda.core.abstract_plugins.components import log_masking
from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

secret = {_SECRET!r}
warnings, seen = [], []


class Warned(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.WARNING and "scrub_credentials" in record.getMessage():
            warnings.append(record.getMessage())


class Seen(logging.Handler):
    def emit(self, record):
        seen.append([record.getMessage(), getattr(record, "tag", None)])


logging.getLogger().addHandler(Warned())
child = logging.getLogger("dis.child")
child.setLevel(logging.DEBUG)
child.addHandler(Seen())
enable_process_wide_log_masking()
with registered_secrets_scope(frozenset([secret])):
    pass
disable_log_masking()
disable_log_masking()
before = (logging.getLogRecordFactory(), logging.Logger.makeRecord)
with registered_secrets_scope(frozenset([secret])):
    child.info("v %s", secret, extra={{"tag": secret}})
after = (logging.getLogRecordFactory(), logging.Logger.makeRecord)
print(json.dumps({{"disabled": log_masking.log_masking_disabled(),
                  "process_wide": log_masking.process_wide_log_masking_enabled(), "seen": seen,
                  "same": before == after, "warnings": len(warnings)}}))
"""


def test_disable_leaves_records_untouched_does_not_reinstall_and_warns_once() -> None:
    result = _run_python(_DISABLE_SCRIPT)

    assert result["disabled"] is True
    assert result["process_wide"] is False
    assert result["seen"] == [[f"v {_SECRET}", _SECRET]]
    assert result["same"] is True
    assert result["warnings"] == 1
