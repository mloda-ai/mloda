from __future__ import annotations

from typing import Any, Callable, TypeVar

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
