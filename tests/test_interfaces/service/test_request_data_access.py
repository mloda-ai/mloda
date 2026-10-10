"""Data-access option keys in a feature request are reported by path, never echoing their value."""

from typing import Any

import pytest

from mloda.user import PluginLoader
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from mloda.interfaces.service.request import Problem, data_access_option_keys, data_access_violations

# PluginLoader.all() can take seconds on a cold import under xdist load.
pytestmark = pytest.mark.timeout(30)

SECRET = "s3://user:pw@host/x"  # nosec B105
KEY = CsvReader.data_access_name()


def _assert_no_secret(problems: list[Problem]) -> None:
    for problem in problems:
        for value in (problem.path, problem.problem, problem.message, problem.suggestion):
            assert value is None or SECRET not in value


def test_option_keys_contain_builtin_file_reader() -> None:
    PluginLoader.all()
    keys = data_access_option_keys()
    assert isinstance(keys, frozenset)
    assert KEY in keys


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
