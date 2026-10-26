"""Service error results: a closed set of codes, a unique audit id, and Problem details."""

import json
import logging
import re

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.validators.missing_input_validator import MissingInputFeatureError
from mloda.interfaces.service.errors import (
    ERROR_CODES,
    MAX_MISSING_INPUTS,
    error_result,
    failure_result,
    is_error,
    problems_to_details,
)
from mloda.interfaces.service.request import MAX_NAME_LENGTH, Problem

AUDIT_ID = re.compile(r"^aud_[0-9a-f]{16,}$")


def _missing(
    group: str = "G", feature: str = "f", inputs: tuple[str, ...] = ("a",), available: tuple[str, ...] = ("secretcol",)
) -> MissingInputFeatureError:
    return MissingInputFeatureError(group, feature, inputs, available)


def _caused_by(error: Exception) -> Exception:
    try:
        try:
            raise error
        except Exception as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as wrapper:
        return wrapper


def _context_of(error: Exception) -> Exception:
    try:
        try:
            raise error
        except Exception:
            raise RuntimeError("wrapper")  # noqa: B904
    except RuntimeError as wrapper:
        return wrapper


def test_error_codes() -> None:
    assert ERROR_CODES == (
        "validation_failed",
        "policy_denied",
        "limit_exceeded",
        "plan_invalid",
        "plan_stale",
        "missing_input_feature",
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


def test_missing_input_feature_code_matches_the_exception() -> None:
    assert "missing_input_feature" in ERROR_CODES
    assert MissingInputFeatureError.code == "missing_input_feature"


@pytest.mark.parametrize("wrap", [lambda e: e, _caused_by], ids=["direct", "cause"])
def test_failure_result_maps_missing_input(wrap: Any) -> None:
    error = _missing("Grp", "feat", ("in1", "in2"))

    result = failure_result("The run", wrap(error))

    body = result["error"]
    assert body["code"] == "missing_input_feature"
    assert "Grp" in body["message"] and "feat" in body["message"]
    assert str(error) not in body["message"]
    assert body["details"] == [{"feature_group": "Grp", "feature": "feat", "missing_inputs": ["in1", "in2"]}]
    assert "available_columns" not in json.dumps(result)
    assert "secretcol" not in json.dumps(result)


def test_failure_result_strips_control_characters_from_names() -> None:
    result = failure_result("The run", _missing("G\x1b[31mr\x00p", "f\x07", ("a\x1b[31mb\x00c",)))

    (detail,) = result["error"]["details"]
    assert detail == {"feature_group": "Grp", "feature": "f", "missing_inputs": ["abc"]}
    assert "\x1b" not in json.dumps(result) and "\\u001b" not in json.dumps(result)


def test_failure_result_removes_tab_newline_and_carriage_return_from_names() -> None:
    result = failure_result("The run", _missing("G\trp", "f\ne", ("a\rb",)))

    (detail,) = result["error"]["details"]
    for value in (detail["feature_group"], detail["feature"], *detail["missing_inputs"]):
        assert not any(char in value for char in "\t\n\r")
    assert detail["feature_group"] in {"G rp", "Grp"}
    assert detail["feature"] in {"f e", "fe"}
    assert detail["missing_inputs"][0] in {"a b", "ab"}


def test_failure_result_truncates_long_names_and_caps_missing_inputs() -> None:
    long = "x" * (MAX_NAME_LENGTH + 50)
    inputs = tuple(f"in{index}" for index in range(MAX_MISSING_INPUTS + 10))

    (detail,) = failure_result("The run", _missing(long, long, (long, *inputs)))["error"]["details"]

    assert detail["feature_group"] == "x" * MAX_NAME_LENGTH
    assert detail["feature"] == "x" * MAX_NAME_LENGTH
    assert len(detail["missing_inputs"]) == MAX_MISSING_INPUTS
    assert detail["missing_inputs"][0] == "x" * MAX_NAME_LENGTH


def test_failure_result_ignores_missing_input_only_in_context() -> None:
    result = failure_result("The run", _context_of(_missing()))
    assert result["error"]["code"] == "internal"
    assert result["error"]["details"] == []


def test_failure_result_other_exception_stays_internal() -> None:
    assert failure_result("The run", ValueError("x"))["error"]["code"] == "internal"


def test_failure_result_logs_the_audit_line_for_missing_input(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="mloda.interfaces.service"):
        result = failure_result("The run", _missing())

    records = [r for r in caplog.records if r.name == "mloda.interfaces.service"]
    assert len(records) == 1
    assert result["error"]["audit_id"] in records[0].getMessage()
