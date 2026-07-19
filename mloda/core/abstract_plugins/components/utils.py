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
    """Like safe_field, but returns (value, None) on success and (fallback, non-empty error, the exception type name if the message is empty) on a raise."""
    try:
        return read(), None
    except catching as exc:
        return fallback, str(exc) or type(exc).__name__


def get_all_subclasses(cls: Any, log_n_subclasses: int = 0) -> set[type[Any]]:
    all_subclasses = set()

    for subclass in cls.__subclasses__():
        all_subclasses.add(subclass)
        all_subclasses.update(get_all_subclasses(subclass))

    if log_n_subclasses > 0:
        logger.debug(f"Abstractclass: {type(cls)}. Subclasses: {list(all_subclasses)[log_n_subclasses]}.")
    return all_subclasses
