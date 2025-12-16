from __future__ import annotations

from typing import Any, Type

import logging

logger = logging.getLogger(__name__)


def get_all_subclasses(cls: Any, log_n_subclasses: int = 0) -> set[Type[Any]]:
    all_subclasses = set()

    for subclass in cls.__subclasses__():
        all_subclasses.add(subclass)
        all_subclasses.update(get_all_subclasses(subclass))

    if log_n_subclasses > 0:
        logger.debug(f"Abstractclass: {type(cls)}. Subclasses: {list(all_subclasses)[log_n_subclasses]}.")
    return all_subclasses
