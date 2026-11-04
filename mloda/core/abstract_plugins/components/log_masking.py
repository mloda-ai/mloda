"""Run-scoped masking of registered secret literals in every log record created inside a run.

Installed lazily on the first non-empty registered_secrets_scope: a LogRecord factory wrapper and a
Logger.makeRecord wrapper. Both pass through when nothing is registered or masking is disabled.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any

from mloda.core.abstract_plugins.components.credential_scrub import (
    _mask_item,
    current_registered_secrets,
    mask_literals,
    redact_exception,
)

DISABLE_LOG_MASKING_COMMAND = "DISABLE_LOG_MASKING"

_UNFORMATTABLE = "<log record masked: unformattable>"
_PROBE = object()
_lock = threading.Lock()
_disabled = False
_warned = False
_installed_factory: Callable[..., logging.LogRecord] | None = None
_installed_make_record: Callable[..., logging.LogRecord] | None = None
_previous_factory: Callable[..., logging.LogRecord] | None = None
_previous_make_record: Callable[..., logging.LogRecord] | None = None
_probe_seen = {"factory": False, "make_record": False}


def _renew_lock_after_fork() -> None:
    global _lock
    _lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_renew_lock_after_fork)


def log_masking_disabled() -> bool:
    return _disabled


def _holds_literal(text: str, literals: frozenset[str]) -> bool:
    return any(literal and literal in text for literal in literals)


def _mask_args(args: Any, literals: frozenset[str]) -> Any:
    if isinstance(args, tuple):
        return tuple(_mask_item(arg, literals) if isinstance(arg, str) else arg for arg in args)
    if isinstance(args, Mapping):
        masked: dict[Any, Any] = {}
        for key in args:
            value = args[key]
            masked[key] = _mask_item(value, literals) if isinstance(value, str) else value
        return masked
    return args


def _mask_extras(record: logging.LogRecord, literals: frozenset[str]) -> None:
    exc_info = record.exc_info
    if isinstance(exc_info, tuple) and len(exc_info) == 3 and isinstance(exc_info[1], BaseException):
        redact_exception(exc_info[1], literals)
    if isinstance(record.stack_info, str):
        record.stack_info = mask_literals(record.stack_info, literals)


def _formatted(msg: Any, args: Any) -> str:
    text = str(msg)
    return text % args if args else text


def _mask_formatted(record: logging.LogRecord, literals: frozenset[str]) -> bool:
    """Mask msg and args in place, keeping the template; False when the result would still leak."""
    msg = record.msg
    if isinstance(msg, str):
        msg = mask_literals(msg, literals)
    elif isinstance(msg, (dict, list, tuple)):
        msg = _mask_item(msg, literals)
    args = _mask_args(record.args, literals)
    if _holds_literal(_formatted(msg, args), literals):
        return False
    record.msg = msg
    record.args = args
    _mask_extras(record, literals)
    return True


def _mask_flattened(record: logging.LogRecord, literals: frozenset[str]) -> bool:
    record.msg = mask_literals(record.getMessage(), literals)
    record.args = ()
    _mask_extras(record, literals)
    return True


def _stringify_args(args: Any, literals: frozenset[str]) -> Any:
    if isinstance(args, tuple):
        return tuple(_masked_text(arg, literals) for arg in args)
    if isinstance(args, Mapping):
        return {key: _masked_text(args[key], literals) for key in args}
    return args


def _masked_text(value: Any, literals: frozenset[str]) -> Any:
    text = str(value)
    return mask_literals(text, literals) if _holds_literal(text, literals) else value


def _mask_unformattable(record: logging.LogRecord, literals: frozenset[str]) -> bool:
    msg = mask_literals(str(record.msg), literals)
    args = _stringify_args(record.args, literals)
    record.msg = msg
    record.args = args
    _mask_extras(record, literals)
    return True


def _mask_blank(record: logging.LogRecord) -> None:
    record.msg = _UNFORMATTABLE
    record.args = ()
    record.exc_info = None
    record.exc_text = None
    record.stack_info = None


def _mask_record(record: logging.LogRecord, literals: frozenset[str]) -> None:
    for strategy in (_mask_formatted, _mask_flattened, _mask_unformattable):
        try:
            done = strategy(record, literals)
        except Exception:  # noqa: BLE001  # a factory must never raise into the caller's log call; fail closed
            done = False
        if done:
            return
    _mask_blank(record)


def _wrap_factory(previous: Callable[..., logging.LogRecord]) -> Callable[..., logging.LogRecord]:
    def masking_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        if record.msg is _PROBE:
            _probe_seen["factory"] = True
            return record
        if not _disabled:
            literals = current_registered_secrets()
            if literals:
                _mask_record(record, literals)
        return record

    return masking_factory


def _wrap_make_record(previous: Callable[..., logging.LogRecord]) -> Callable[..., logging.LogRecord]:
    def masking_make_record(
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
        record = previous(self, name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)
        if msg is _PROBE:
            _probe_seen["make_record"] = True
            return record
        if extra and not _disabled:
            literals = current_registered_secrets()
            if literals:
                for key in extra:
                    value = record.__dict__[key] if key in record.__dict__ else None
                    if isinstance(value, str):
                        record.__dict__[key] = mask_literals(value, literals)
        return record

    return masking_make_record


def _factory_in_chain() -> bool:
    _probe_seen["factory"] = False
    logging.getLogRecordFactory()("probe", logging.INFO, "", 0, _PROBE, (), None)
    return _probe_seen["factory"]


def _make_record_in_chain() -> bool:
    _probe_seen["make_record"] = False
    probe_logger = logging.Logger("mloda.log_masking.probe")
    logging.Logger.makeRecord(probe_logger, "probe", logging.INFO, "", 0, _PROBE, (), None)
    return _probe_seen["make_record"]


def _installed() -> bool:
    return logging.getLogRecordFactory() is _installed_factory and logging.Logger.makeRecord is _installed_make_record


def ensure_log_masking() -> None:
    """Install the factory and makeRecord wrappers unless they are already in the chain or masking is disabled."""
    global _installed_factory, _installed_make_record, _previous_factory, _previous_make_record
    if _disabled or _installed():
        return
    with _lock:
        if _disabled:
            return
        if not _factory_in_chain():
            _previous_factory = logging.getLogRecordFactory()
            _installed_factory = _wrap_factory(_previous_factory)
            logging.setLogRecordFactory(_installed_factory)
        if not _make_record_in_chain():
            _previous_make_record = logging.Logger.makeRecord
            _installed_make_record = _wrap_make_record(_previous_make_record)
            setattr(logging.Logger, "makeRecord", _installed_make_record)


def _switch_off() -> bool:
    """Disable masking and unwind our wrappers when outermost; return True on the first call."""
    global _disabled, _installed_factory, _installed_make_record
    with _lock:
        first = not _disabled
        _disabled = True
        if _installed_factory is not None and logging.getLogRecordFactory() is _installed_factory:
            logging.setLogRecordFactory(_previous_factory or logging.LogRecord)
            _installed_factory = None
        if _installed_make_record is not None and logging.Logger.makeRecord is _installed_make_record:
            setattr(logging.Logger, "makeRecord", _previous_make_record)
            _installed_make_record = None
    return first


def disable_log_masking_quietly() -> None:
    _switch_off()


def disable_log_masking() -> None:
    """Turn run-scoped log masking off for this process; plugin log lines are then the provider's duty."""
    global _warned
    _switch_off()
    if not _warned:
        _warned = True
        logging.getLogger(__name__).warning(
            "Log record masking is disabled: plugin log lines are no longer masked; "
            "providers must use scrub_credentials on anything they log"
        )
