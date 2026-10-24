"""Data-access option keys in a feature request are reported by path, never echoing their value."""

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import RESERVED_READER_OPTION_KEY
from mloda.core.abstract_plugins.components.input_data.source_feature_group import concrete_source_groups
from mloda.core.abstract_plugins.components.input_data.read_db_fg import QUERY_OPTION
from mloda.provider import ApiDataFG, DataCreator
from mloda.user import PluginLoader
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from mloda.interfaces.service.request import Problem, data_access_option_keys, data_access_violations

# PluginLoader.all() can take seconds on a cold import under xdist load.
pytestmark = pytest.mark.timeout(30)

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


def test_option_keys_contain_base_input_data_names() -> None:
    PluginLoader.all()
    assert DataCreator.data_access_name() in data_access_option_keys()


def test_option_keys_contain_api_data_pointer_key() -> None:
    PluginLoader.all()
    assert ApiDataFG.__name__ in data_access_option_keys()


def test_option_keys_contain_reserved_reader_key() -> None:
    """mloda stores the chosen reader and its data access under this key, so a request must not set it."""
    assert RESERVED_READER_OPTION_KEY == "BaseInputData"
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
