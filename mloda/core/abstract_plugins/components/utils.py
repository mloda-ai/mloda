from __future__ import annotations

from typing import Any, Callable, TypeVar

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_field(read: Callable[[], T], fallback: T, catching: tuple[type[Exception], ...] = (Exception,)) -> T:
    """Annotate tier: degrade a single unreadable field to a fallback instead of failing the whole discovery call."""
    try:
        return read()
    except catching:
        return fallback


def get_all_subclasses(cls: Any, log_n_subclasses: int = 0) -> set[type[Any]]:
    all_subclasses = set()

    for subclass in cls.__subclasses__():
        all_subclasses.add(subclass)
        all_subclasses.update(get_all_subclasses(subclass))

    if log_n_subclasses > 0:
        logger.debug(f"Abstractclass: {type(cls)}. Subclasses: {list(all_subclasses)[log_n_subclasses]}.")
    return all_subclasses
