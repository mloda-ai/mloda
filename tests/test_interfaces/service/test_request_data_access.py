"""Data-access option keys in a feature request are reported by path, never echoing their value."""

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import RESERVED_READER_OPTION_KEY
from mloda.core.abstract_plugins.components.input_data.source_feature_group import concrete_source_groups
from mloda.core.abstract_plugins.components.input_data.read_db_fg import QUERY_OPTION
from mloda.provider import ApiDataFG, property_spec
from mloda.user import PluginLoader
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from mloda.interfaces.service.request import (
    Problem,
    caller_refused_option_keys,
    data_access_option_keys,
    data_access_violations,
    refused_group_option_violations,
    refused_option_violations,
)
from tests.test_interfaces.source_groups import CALLER_LOCKED_KEY, CallerLockedMcp6  # noqa: F401


SECRET = "s3://user:pw@host/x"  # nosec B105
KEY = CsvFG.data_access_name()


def _assert_no_secret(problems: list[Problem]) -> None:
    for problem in problems:
        for value in (problem.path, problem.problem, problem.message, problem.suggestion):
            assert value is None or SECRET not in value


def test_csv_pointer_key_is_the_group_name() -> None:
    assert KEY == "CsvFG"


def test_option_keys_contain_builtin_file_reader() -> None:
    PluginLoader.all()
    keys = data_access_option_keys()
    assert isinstance(keys, frozenset)
    assert KEY in keys


@pytest.mark.parametrize("key", ["CsvFG", "JsonFG", "ParquetFG"])
def test_option_keys_contain_stock_source_group_pointers(key: str) -> None:
    PluginLoader.all()
    assert key in data_access_option_keys()


def test_option_keys_contain_every_loaded_source_group_pointer_key() -> None:
    PluginLoader.all()
    keys = data_access_option_keys()
    groups = concrete_source_groups()
    assert groups
    missing = sorted({pointer for group in groups.values() for pointer in group.pointer_keys()} - keys)
    assert missing == []


def test_option_keys_contain_query_option() -> None:
    PluginLoader.all()
    assert QUERY_OPTION == "query_text"
    assert QUERY_OPTION in data_access_option_keys()


def test_option_keys_contain_api_data_pointer_key() -> None:
    PluginLoader.all()
    assert ApiDataFG.__name__ in data_access_option_keys()


def test_option_keys_contain_reserved_reader_key() -> None:
    """mloda stores the chosen reader and its data access under this key, so a request must not set it."""
    assert RESERVED_READER_OPTION_KEY == "mloda_source_reader"
    assert RESERVED_READER_OPTION_KEY in data_access_option_keys()


def test_reserved_reader_key_is_reported() -> None:
    request = {"features": [{"name": "x", "options": {RESERVED_READER_OPTION_KEY: "anything"}}]}

    problems = data_access_violations(request)

    assert [p.path for p in problems] == [f"/features/0/options/{RESERVED_READER_OPTION_KEY}"]
    assert problems[0].problem == "data_access_option"
    assert all("anything" not in p.message for p in problems)


def test_csv_pointer_at_system_file_is_reported_without_value() -> None:
    PluginLoader.all()
    request = {"features": [{"name": "x", "options": {"CsvFG": "/etc/passwd"}}]}

    problems = data_access_violations(request)

    assert [p.path for p in problems] == ["/features/0/options/CsvFG"]
    assert problems[0].problem == "data_access_option"
    for problem in problems:
        for value in (problem.path, problem.problem, problem.message, problem.suggestion):
            assert value is None or "/etc/passwd" not in value


def test_query_option_is_reported() -> None:
    PluginLoader.all()
    request = {"features": [{"name": "x", "options": {QUERY_OPTION: "SELECT 1"}}]}

    problems = data_access_violations(request)

    assert [p.path for p in problems] == [f"/features/0/options/{QUERY_OPTION}"]
    assert all("SELECT" not in p.message for p in problems)


def test_clean_request_has_no_violations() -> None:
    request = {"features": ["a", {"name": "b", "options": {"k": 1}, "in_features": ["x"]}]}
    assert data_access_violations(request) == []


def test_string_features_are_ignored() -> None:
    assert data_access_violations({"features": [KEY]}) == []


@pytest.mark.parametrize("container", ["options", "group_options", "context_options"])
def test_key_in_container_is_reported(container: str) -> None:
    PluginLoader.all()
    request: dict[str, Any] = {"features": ["a", {"name": "b", container: {"k": 1, KEY: SECRET}}]}

    problems = data_access_violations(request)

    assert [p.path for p in problems] == [f"/features/1/{container}/{KEY}"]
    assert problems[0].problem == "data_access_option"
    assert KEY in problems[0].message
    _assert_no_secret(problems)


def test_key_in_nested_in_features_is_reported() -> None:
    PluginLoader.all()
    nested = {"name": "src", "options": {"in_features": {"name": "raw", "options": {KEY: SECRET}}}}
    request = {"features": [{"name": "b", "options": {"in_features": nested}}]}

    problems = data_access_violations(request)

    assert [p.path for p in problems] == [f"/features/0/options/in_features/options/in_features/options/{KEY}"]
    assert problems[0].problem == "data_access_option"
    _assert_no_secret(problems)


def test_key_in_sibling_in_features_of_nested_dict_is_reported() -> None:
    PluginLoader.all()
    inner = {"name": "src", "in_features": {"name": "raw", "options": {KEY: SECRET}}}
    request = {"features": [{"name": "b", "options": {"in_features": inner}}]}

    problems = data_access_violations(request)

    assert [p.path for p in problems] == [f"/features/0/options/in_features/in_features/options/{KEY}"]
    assert problems[0].problem == "data_access_option"
    _assert_no_secret(problems)


@pytest.mark.parametrize("value", [5, [1, None], {"no_name": 1}, {"name": "src", "in_features": 5}])
def test_odd_in_features_shapes_report_nothing(value: Any) -> None:
    PluginLoader.all()
    request = {"features": [{"name": "b", "options": {"in_features": value}}]}

    assert data_access_violations(request) == []


def test_key_in_nested_in_features_of_group_options_is_reported() -> None:
    PluginLoader.all()
    request = {"features": [{"name": "b", "group_options": {"in_features": {"name": "raw", "options": {KEY: SECRET}}}}]}

    problems = data_access_violations(request)

    assert [p.path for p in problems] == [f"/features/0/group_options/in_features/options/{KEY}"]
    _assert_no_secret(problems)


def test_every_violation_is_reported() -> None:
    PluginLoader.all()
    request = {
        "features": [
            {"name": "a", "options": {KEY: SECRET}},
            {"name": "b", "context_options": {KEY: SECRET}},
        ]
    }

    problems = data_access_violations(request)

    assert sorted(p.path for p in problems) == [f"/features/0/options/{KEY}", f"/features/1/context_options/{KEY}"]
    _assert_no_secret(problems)


def test_caller_refused_keys_contain_the_flagged_key() -> None:
    keys = caller_refused_option_keys()
    assert isinstance(keys, frozenset)
    assert CALLER_LOCKED_KEY in keys
    assert KEY not in keys


@pytest.mark.parametrize("container", ["options", "group_options", "context_options"])
def test_refused_key_in_container_is_reported_without_value(container: str) -> None:
    request: dict[str, Any] = {"features": ["a", {"name": "b", container: {"k": 1, CALLER_LOCKED_KEY: SECRET}}]}

    problems = refused_option_violations(request)

    assert [p.path for p in problems] == [f"/features/1/{container}/{CALLER_LOCKED_KEY}"]
    assert problems[0].problem == "option_not_caller_settable"
    _assert_no_secret(problems)


def test_refused_key_in_nested_in_features_is_reported() -> None:
    nested = {"name": "raw", "options": {CALLER_LOCKED_KEY: SECRET}}
    request = {"features": [{"name": "b", "options": {"in_features": nested}}]}

    problems = refused_option_violations(request)

    assert [p.path for p in problems] == [f"/features/0/options/in_features/options/{CALLER_LOCKED_KEY}"]
    _assert_no_secret(problems)


def test_clean_or_malformed_request_has_no_refused_keys() -> None:
    assert refused_option_violations({"features": ["a", {"name": "b", "options": {"k": 1}}]}) == []
    assert refused_option_violations({"features": "a"}) == []
    assert refused_option_violations(None) == []


def test_refused_group_option_is_reported_by_pointer() -> None:
    problems = refused_group_option_violations({"k": 1, CALLER_LOCKED_KEY: SECRET})

    assert [p.path for p in problems] == [f"/options/{CALLER_LOCKED_KEY}"]
    assert problems[0].problem == "option_not_caller_settable"
    _assert_no_secret(problems)


def test_malformed_property_mapping_does_not_break_key_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    not_a_mapping = type("NotAMapping", (), {"PROPERTY_MAPPING": ["mcp_malformed_a"]})
    non_spec_values = type("NonSpecValues", (), {"PROPERTY_MAPPING": {"mcp_malformed_b": "just text"}})
    flagged = type(
        "Flagged",
        (),
        {"PROPERTY_MAPPING": {"mcp_flagged": property_spec("Locked.", default=None, caller_settable=False)}},
    )
    monkeypatch.setattr(
        "mloda.interfaces.service.request.feature_group_subclasses",
        lambda: [not_a_mapping, non_spec_values, flagged],
    )

    assert caller_refused_option_keys() == frozenset({"mcp_flagged"})
