"""Service error results: a closed set of codes, a unique audit id per error, and Problem details."""

import logging
import re
import secrets
from collections.abc import Iterable, Mapping
from typing import Any

from mloda.core.abstract_plugins.components.utils import failure_report
from mloda.interfaces.service.request import Problem

_LOGGER = logging.getLogger("mloda.interfaces.service")

ERROR_CODES = (
    "validation_failed",
    "policy_denied",
    "limit_exceeded",
    "plan_invalid",
    "plan_stale",
    "internal",
)
# ANSI escape sequences, then every C0/C1 control character except tab, newline and carriage return.
CONTROL_CHARACTERS = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def strip_control_characters(text: str) -> str:
    """Remove ANSI escape sequences and control characters other than tab, newline and carriage return."""
    return CONTROL_CHARACTERS.sub("", text)


def error_result(
    code: str, message: str, details: Iterable[Mapping[str, Any]] = (), **extra: Any
) -> dict[str, dict[str, Any]]:
    """Build an error result with a fresh audit id; extra keys are merged into the error object."""
    if code not in ERROR_CODES:
        raise ValueError(f"Unknown error code {code!r}; known: {list(ERROR_CODES)}")
    error = {
        **extra,
        "code": code,
        "message": strip_control_characters(message),
        "details": [dict(detail) for detail in details],
        "audit_id": f"aud_{secrets.token_hex(8)}",
    }
    return {"error": error}


def failure_result(action: str, error: Exception) -> dict[str, dict[str, Any]]:
    """An internal error naming only the exception class; the log gets the audit id and a traceback without messages."""
    result = error_result("internal", f"{action} failed with {type(error).__name__}.")
    _LOGGER.error("%s failed audit_id=%s\n%s", action, result["error"]["audit_id"], failure_report(error)[1])
    return result


def problems_to_details(problems: Iterable[Problem]) -> list[dict[str, str]]:
    """Convert Problems to error details, omitting an absent suggestion."""
    details = []
    for problem in problems:
        detail = {"path": problem.path, "problem": problem.problem, "message": problem.message}
        if problem.suggestion is not None:
            detail["suggestion"] = problem.suggestion
        details.append(detail)
    return details


def is_error(result: Mapping[str, Any]) -> bool:
    """Is this result an error result?"""
    return "error" in result
