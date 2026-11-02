"""Declarations for switching a conformance test off, and key validation."""

import difflib
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import Enum


class DeclarationError(Exception):
    """Raised when a switch-off declaration is invalid."""


class Cause(Enum):
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    BACKEND = "BACKEND"
    ENVIRONMENT = "ENVIRONMENT"
    OTHER = "OTHER"


@dataclass(frozen=True)
class SwitchOff:
    """Why a contract test is switched off."""

    cause: Cause
    reason: str = ""
    docs: str = ""

    def __post_init__(self) -> None:
        if self.cause is not Cause.NOT_IMPLEMENTED and not self.reason:
            raise DeclarationError(f"SwitchOff({self.cause.name}) requires a reason")
        if self.cause is Cause.OTHER and not self.docs:
            raise DeclarationError("SwitchOff(OTHER) requires docs")


def unknown_key_failures(declared: Iterable[str], known: Collection[str], where: str) -> tuple[str, ...]:
    """Return one message per declared key not in known, with the closest match."""
    failures: list[str] = []
    for key in declared:
        if key in known:
            continue
        close = difflib.get_close_matches(key, known, n=1)
        hint = f"; did you mean '{close[0]}'?" if close else ""
        failures.append(f"{where}: unknown key '{key}'{hint}")
    return tuple(failures)
