"""Service error results: a closed set of codes, a unique audit id, and Problem details."""

import json
import re

import pytest

from mloda.interfaces.service.errors import ERROR_CODES, error_result, is_error, problems_to_details
from mloda.interfaces.service.request import Problem

AUDIT_ID = re.compile(r"^aud_[0-9a-f]{16,}$")


def test_error_codes() -> None:
    assert ERROR_CODES == (
        "validation_failed",
        "policy_denied",
        "limit_exceeded",
        "plan_invalid",
        "plan_stale",
        "internal",
    )


@pytest.mark.parametrize("code", ERROR_CODES)
def test_error_result_shape(code: str) -> None:
    result = error_result(code, "Something went wrong.")

    assert set(result) == {"error"}
    error = result["error"]
    assert set(error) == {"code", "message", "details", "audit_id"}
    assert error["code"] == code
    assert error["message"] == "Something went wrong."
    assert error["details"] == []
    assert AUDIT_ID.match(error["audit_id"])
    json.dumps(result)


def test_error_result_unknown_code_raises() -> None:
    with pytest.raises(ValueError):
        error_result("not_a_code", "m")


def test_audit_id_is_unique_per_call() -> None:
    ids = {error_result("internal", "m")["error"]["audit_id"] for _ in range(50)}
    assert len(ids) == 50


def test_error_result_carries_details_as_list_of_dicts() -> None:
    details = ({"path": "/name", "problem": "unknown_feature", "message": "m"},)

    error = error_result("validation_failed", "m", details)["error"]

    assert error["details"] == [{"path": "/name", "problem": "unknown_feature", "message": "m"}]
    assert isinstance(error["details"], list)


def test_error_result_merges_extra_keys() -> None:
    error = error_result("policy_denied", "m", alternatives=["a", "b"])["error"]
    assert error["alternatives"] == ["a", "b"]
    assert error["code"] == "policy_denied"


def test_error_result_strips_ansi_escape_from_message() -> None:
    error = error_result("internal", "\x1b[31mboom\x1b[0m")["error"]
    assert error["message"] == "boom"


def test_error_result_strips_control_characters_from_message() -> None:
    error = error_result("internal", "a\x00b\x07c")["error"]
    assert error["message"] == "abc"


def test_problems_to_details_without_suggestion() -> None:
    details = problems_to_details([Problem(path="/a", problem="invalid", message="bad")])
    assert details == [{"path": "/a", "problem": "invalid", "message": "bad"}]


def test_problems_to_details_with_suggestion() -> None:
    problems = [
        Problem(path="/a", problem="unknown_source", message="m1", suggestion="sales"),
        Problem(path="/b", problem="invalid", message="m2"),
    ]

    details = problems_to_details(problems)

    assert details == [
        {"path": "/a", "problem": "unknown_source", "message": "m1", "suggestion": "sales"},
        {"path": "/b", "problem": "invalid", "message": "m2"},
    ]


def test_problems_to_details_empty() -> None:
    assert problems_to_details([]) == []


def test_is_error() -> None:
    assert is_error(error_result("internal", "m")) is True
    assert is_error({"items": [], "next_cursor": None}) is False
    assert is_error({"error": None}) is True
