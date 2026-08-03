from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")

E = TypeVar("E", bound=BaseException)

# Provenance marker for a framework-owned raise. Not an exception type: the object must stay exactly as raised.
MATCH_ABORT_FLAG = "_mloda_match_abort"

# Exception classes a user callable raises when it merely cannot judge a value.
_EXPECTED_JUDGMENT_ERRORS: tuple[type[Exception], ...] = (TypeError, ValueError, AttributeError)


def contained_raise_log_level(exc: BaseException) -> int:
    """DEBUG for expected judgment failures, WARNING for classes that suggest a broken callable."""
    return logging.DEBUG if isinstance(exc, _EXPECTED_JUDGMENT_ERRORS) else logging.WARNING


def escalate_match_abort(exc: E) -> E:
    """Mark a framework-owned raise so the match seam re-raises it instead of containing it as a non-match.

    Mark-or-contain policy: see IdentifyFeatureGroupClass._filter_feature_group_by_criteria.
    """
    # __dict__, not setattr: setattr raises on a frozen-dataclass exception, and failing to mark must not
    # replace the exception being marked.
    try:
        exc.__dict__[MATCH_ABORT_FLAG] = True
    except Exception:  # noqa: BLE001  (marking is never worth losing the original raise)
        logger.debug("Could not mark %s as a match abort.", type(exc).__name__)
    return exc


def is_match_abort(exc: BaseException) -> bool:
    """Is this raise marked as framework-owned, so the match seam must not contain it."""
    # __dict__, not getattr: a custom __getattr__ could raise inside the seam's except block or fake the marker.
    return exc.__dict__.get(MATCH_ABORT_FLAG, False) is True


def safe_field(
    read: Callable[[], T],
    fallback: T,
    catching: tuple[type[Exception], ...] = (Exception,),
    field: str = "",
) -> T:
    """Annotate tier: degrade a single unreadable field to a fallback instead of failing the whole discovery call.

    A labelled read (non-empty `field`) warns on swallow; an unlabelled read degrades silently, because degrading
    there is expected.
    """
    try:
        return read()
    except catching as exc:
        if field:
            # str(exc), not exc: a retained log record must not pin the traceback, its frames and the plugin class.
            logger.warning("Degraded field '%s': %s: %s", field, type(exc).__name__, str(exc))
        return fallback


def contained_raise_reason(exc: BaseException) -> str:
    """Text form of a contained raise: type and message, never the exception object."""
    # partial, not a lambda: exc binds eagerly, so no closure keeps it and its traceback alive.
    return f"raised {type(exc).__name__}: {safe_field(functools.partial(str, exc), type(exc).__name__)}"


def safe_field_with_error(
    read: Callable[[], T],
    fallback: T,
    catching: tuple[type[Exception], ...] = (Exception,),
) -> tuple[T, str | None]:
    """Like safe_field but returns (value, None), else (fallback, str(exc) or the exception type name when blank)."""
    try:
        return read(), None
    except catching as exc:
        message = str(exc)
        return fallback, message if message.strip() else type(exc).__name__


def as_str(value: Any) -> str:
    """Return `value` unchanged, raising TypeError on a non-str so the guarded read that wraps it degrades."""
    if not isinstance(value, str):
        raise TypeError(f"expected str, got {type(value).__name__}")
    return value


def get_all_subclasses(cls: Any) -> set[type[Any]]:
    all_subclasses = set()

    for subclass in cls.__subclasses__():
        all_subclasses.add(subclass)
        all_subclasses.update(get_all_subclasses(subclass))

    return all_subclasses
