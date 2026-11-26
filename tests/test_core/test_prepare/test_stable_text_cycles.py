"""Planning reaches `stable_text` with every option value; a self-referential one must not crash it."""

from typing import Any

import pytest

from mloda.core.prepare.stable_text import stable_text
from mloda.provider import DataCreatorFeatureGroup, FeatureSet
from mloda.user import Feature, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


def _self_referential_list() -> list[Any]:
    cyclic: list[Any] = [1]
    cyclic.append(cyclic)
    return cyclic


def _self_referential_dict() -> dict[str, Any]:
    cyclic: dict[str, Any] = {"a": 1}
    cyclic["self"] = cyclic
    return cyclic


def _options_in_own_group() -> Options:
    o = Options(group={})
    o.group["self"] = o
    return o


def _set_through_options() -> Options:
    s: set[Any] = set()
    o = Options(group={"s": s})
    s.add(o)
    return o


class CyclicOptionSource(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"cyclic_option_source"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"cyclic_option_source": [1]}


def test_a_self_referential_list_is_written_like_repr() -> None:
    assert stable_text(_self_referential_list()) == "[1, [...]]"


def test_a_self_referential_dict_is_written_like_repr() -> None:
    assert stable_text(_self_referential_dict()) == "{'a': 1, 'self': {...}}"


def test_a_cycle_through_a_tuple_and_a_dict_inside_options_has_a_text() -> None:
    inner: list[Any] = []
    outer = (inner,)
    inner.append({"back": outer})

    text = stable_text(Options(group={"g": outer}))

    assert text == "Options(group={'g': ([{'back': (...)}],)}, context={})"


def test_a_value_seen_twice_without_a_cycle_is_written_in_full_both_times() -> None:
    shared = [1, 2]

    assert stable_text([shared, shared]) == "[[1, 2], [1, 2]]"
    assert stable_text({"x": shared, "y": (shared,)}) == "{'x': [1, 2], 'y': ([1, 2],)}"


@pytest.mark.parametrize(
    "build, expected",
    [
        (_options_in_own_group, "Options(group={'self': Options(...)}, context={})"),
        (_set_through_options, "Options(group={'s': {Options(...)}}, context={})"),
    ],
    ids=["options_in_own_group", "set_through_options"],
)
def test_a_cycle_closing_through_options_has_a_text(build: Any, expected: str) -> None:
    assert stable_text(build()) == expected


@pytest.mark.parametrize("build", [_self_referential_list, _self_referential_dict], ids=["list", "dict"])
def test_planning_a_request_with_a_self_referential_context_value_does_not_recurse(build: Any) -> None:
    feature = Feature("cyclic_option_source", Options(context={"c": build()}))

    results = mloda.run_all(
        [feature],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({CyclicOptionSource}),
        parallelization_modes={ParallelizationMode.SYNC},
    )

    assert [list(frame["cyclic_option_source"]) for frame in results] == [[1]]
