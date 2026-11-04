"""Record-level and install/chain/disable tests for run-scoped log record masking.

Inside a non-empty registered_secrets_scope every log record has the registered literals masked.
Process-global changes that cannot be undone in-process run in a subprocess.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess  # nosec B404
import sys
import threading
import traceback
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, cast
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components import log_masking
from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

_SECRET = "Zk8-secret-Vq3LmRt"  # nosec B105
_OTHER = "Pw5-other-Hy2NcXs"  # nosec B105
_STATE = (("_disabled", False), ("_installed_factory", None), ("_installed_make_record", None))


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
    saved_state = {name: getattr(log_masking, name, None) for name, _ in _STATE}
    logging.setLogRecordFactory(logging.LogRecord)
    setattr(logging.Logger, "makeRecord", _plain_make_record)
    for name, value in _STATE:
        setattr(log_masking, name, value)
    yield
    logging.setLogRecordFactory(saved_factory)
    setattr(logging.Logger, "makeRecord", saved_make_record)
    for name, value in saved_state.items():
        setattr(log_masking, name, value)


class _Capture(logging.Handler):
    """Collects records without formatting them."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []
        self.logger = logging.getLogger(f"lm_test.{uuid4().hex}.child")
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


def _assert_no_literal(record: logging.LogRecord, literal: str) -> None:
    assert literal not in record.getMessage()
    assert literal not in str(record.msg)
    assert literal not in repr(record.args)


_NO_LEAK_CASES = [
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


@pytest.mark.parametrize("literal, msg, args", _NO_LEAK_CASES)
def test_registered_literal_never_survives_in_a_record(
    capture: _Capture, literal: str, msg: Any, args: tuple[Any, ...]
) -> None:
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


@pytest.mark.parametrize("wrap", [dict, MappingProxyType], ids=["dict", "mappingproxy"])
def test_mapping_arg_masks_values_and_keeps_keys(capture: _Capture, wrap: Any) -> None:
    mapping = wrap({"user": _SECRET, "other": "x"})
    with _scope(_SECRET):
        capture.logger.info("u=%(user)s o=%(other)s", mapping)

    (record,) = capture.records
    assert record.getMessage() == "u=*** o=x"
    assert isinstance(record.args, Mapping)
    assert set(record.args) == {"user", "other"}


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
    "msg, args",
    [
        pytest.param("a %s %s", (_SECRET,), id="wrong_count"),
        pytest.param("%(a)s %(b)s", ({"a": _SECRET},), id="missing_key"),
        pytest.param(f"x {_SECRET}", ("extra",), id="no_placeholder"),
        pytest.param("x %s %s", (_Boom(), _SECRET), id="raising_str"),
    ],
)
def test_malformed_call_is_masked_and_never_raises(capture: _Capture, msg: str, args: tuple[Any, ...]) -> None:
    with _scope(_SECRET):
        capture.logger.info(msg, *args)

    (record,) = capture.records
    assert _SECRET not in str(record.msg)
    assert _SECRET not in repr(record.args)


def test_arg_with_raising_str_does_not_make_the_log_call_raise(capture: _Capture) -> None:
    with _scope(_SECRET):
        capture.logger.info("x %s", _Boom())

    assert len(capture.records) == 1


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


def test_str_extra_values_are_masked_and_other_values_untouched(capture: _Capture) -> None:
    holder = _Holder()
    with _scope(_SECRET):
        capture.logger.info("m", extra={"tag": _SECRET, "count": 5, "obj": holder})

    (record,) = capture.records
    assert _SECRET not in getattr(record, "tag")
    assert getattr(record, "count") == 5
    assert getattr(record, "obj") is holder


@pytest.mark.parametrize("installed", [False, True], ids=["never_installed", "installed_before"])
def test_records_are_untouched_with_nothing_registered(capture: _Capture, installed: bool) -> None:
    if installed:
        with _scope(_SECRET):
            pass
    capture.logger.info("login %s", _SECRET, extra={"tag": _SECRET})

    (record,) = capture.records
    assert record.args == (_SECRET,)
    assert getattr(record, "tag") == _SECRET


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


def test_record_from_a_thread_outside_any_scope_is_untouched(capture: _Capture) -> None:
    entered, release = threading.Event(), threading.Event()

    def hold_scope() -> None:
        with _scope(_SECRET):
            entered.set()
            release.wait(5)

    holder = threading.Thread(target=hold_scope)
    holder.start()
    assert entered.wait(5)
    outside = threading.Thread(target=lambda: capture.logger.info("raw %s", _SECRET))
    outside.start()
    outside.join(5)
    release.set()
    holder.join(5)

    (record,) = capture.records
    assert record.getMessage() == f"raw {_SECRET}"


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


def test_no_registered_literal_reaches_the_regex_cache(capture: _Capture) -> None:
    literal = f"lit-{uuid4().hex}"
    with _scope(literal):
        capture.logger.info("x %s", literal)
        capture.logger.info(f"pre {literal}")

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


@pytest.mark.skipif(sys.version_info < (3, 14), reason="thread_inherit_context needs Python 3.14")
def test_pool_thread_with_a_stale_set_is_masked_against_it() -> None:
    assert _run_python(_INHERIT_SCRIPT, "-X", "thread_inherit_context=1") == ["v ***"]


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
from mloda.steward import disable_log_masking
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
with registered_secrets_scope(frozenset([secret])):
    pass
disable_log_masking()
disable_log_masking()
before = (logging.getLogRecordFactory(), logging.Logger.makeRecord)
with registered_secrets_scope(frozenset([secret])):
    child.info("v %s", secret, extra={{"tag": secret}})
after = (logging.getLogRecordFactory(), logging.Logger.makeRecord)
print(json.dumps({{"disabled": log_masking.log_masking_disabled(), "seen": seen,
                  "same": before == after, "warnings": len(warnings)}}))
"""


def test_disable_leaves_records_untouched_does_not_reinstall_and_warns_once() -> None:
    result = _run_python(_DISABLE_SCRIPT)

    assert result["disabled"] is True
    assert result["seen"] == [[f"v {_SECRET}", _SECRET]]
    assert result["same"] is True
    assert result["warnings"] == 1


def test_masking_is_enabled_by_default() -> None:
    assert log_masking.log_masking_disabled() is False
