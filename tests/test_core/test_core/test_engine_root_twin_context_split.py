"""Root copies reached with differing pushed context compute once per context in separate frames (#2773).

Consumers push ``context={"k": ...}`` onto a root input without propagate/cascade keys.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_abstract_plugins.name_bound_fixtures import make_root_fg
from tests.test_plugins.integration_plugins.chainer.chainer_context_feature import ChainedContextFeatureGroupTest
from tests.test_plugins.integration_plugins.chainer.chainer_test_feature import (
    ChainedFeatureGroupTest,
    ChainedFeatureGroupTest_B,
)

ROOT = "rsrc"
SENTINEL = -1


def _values(data: Any) -> list[Any]:
    column = data[ROOT]
    if isinstance(data, dict):
        return list(column)
    return list(column.to_pylist())


def _consumer_groups(prefix: str, names_to_k: dict[str, int], one_group: bool) -> list[type[FeatureGroup]]:
    """Consumer groups copying ``rsrc`` read with ``context={"k": k}`` per consumer name."""

    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        return {str(feature.name): _values(data) for feature in features.features}

    def make(name: str, names: set[str]) -> type[FeatureGroup]:
        def feature_names_supported(cls: Any) -> set[str]:
            return set(names)

        def input_features(self: Any, options: Options, feature_name: FeatureName) -> set[Feature]:
            return {Feature(ROOT, Options(context={"k": names_to_k[str(feature_name)]}))}

        namespace = {
            "__module__": __name__,
            "feature_names_supported": classmethod(feature_names_supported),
            "input_features": input_features,
            "calculate_feature": classmethod(calculate_feature),
        }
        return type(f"{prefix}{name}ConsumerFeatureGroup", (FeatureGroup,), namespace)

    if one_group:
        return [make("One", set(names_to_k))]
    return [make(name.capitalize(), {name}) for name in names_to_k]


def _context_root(prefix: str, calls: list[list[Any]]) -> type[FeatureGroup]:
    """Dict-returning root writing ``k * 10`` per feature, or SENTINEL when ``k`` is absent."""

    def feature_names_supported(cls: Any) -> set[str]:
        return {ROOT}

    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        calls.append([feature.options.get("k") for feature in features.features])
        out: dict[str, list[Any]] = {}
        for feature in features.features:
            k = feature.options.get("k")
            out[str(feature.name)] = [SENTINEL if k is None else k * 10]
        return out

    namespace = {
        "__module__": __name__,
        "feature_names_supported": classmethod(feature_names_supported),
        "calculate_feature": classmethod(calculate_feature),
    }
    return type(f"{prefix}ContextRootFeatureGroup", (DataCreatorFeatureGroup,), namespace)


def _columns(results: list[Any]) -> dict[str, list[Any]]:
    """Map every result column name to its values, across all result frames."""
    out: dict[str, list[Any]] = {}
    for result in results:
        table = result.to_pydict() if isinstance(result, pa.Table) else result
        for name, values in table.items():
            out.setdefault(str(name), []).append(list(values))
    return out


SHAPES = {
    "two_groups": ({"c1": 1, "c2": 2}, False, []),
    "one_group": ({"c1": 1, "c2": 2}, True, []),
    "consumer_plus_plain": ({"c1": 1}, False, [ROOT]),
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_append_root_computes_once_per_pushed_context(shape: str) -> None:
    names_to_k, one_group, extra = SHAPES[shape]
    calls: list[list[str]] = []
    prefix = f"TwinSplitAppend{shape.title().replace('_', '')}"
    root = make_root_fg(prefix, __name__, ROOT, [1, 2], append_columns=True, calls=calls)
    consumers = _consumer_groups(prefix, names_to_k, one_group)

    results = mloda.run_all(
        [*names_to_k, *extra],
        compute_frameworks=[PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups({root, *consumers}),
    )

    assert calls == [[ROOT], [ROOT]]
    columns = _columns(results)
    for name in names_to_k:
        assert columns[name] == [[1, 2]]
    if extra:
        assert columns[ROOT] == [[1, 2]]


@pytest.mark.parametrize("framework", [PythonDictFramework, PyArrowTable], ids=["python_dict", "pyarrow"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_dict_root_keeps_each_pushed_context_value(shape: str, framework: Any) -> None:
    names_to_k, one_group, extra = SHAPES[shape]
    calls: list[list[Any]] = []
    prefix = f"TwinSplitDict{shape.title().replace('_', '')}{framework.__name__}"
    root = _context_root(prefix, calls)
    consumers = _consumer_groups(prefix, names_to_k, one_group)

    results = mloda.run_all(
        [*names_to_k, *extra],
        compute_frameworks=[framework],
        plugin_collector=PluginCollector.enabled_feature_groups({root, *consumers}),
    )

    expected_calls = sorted([[k] for k in names_to_k.values()] + [[None] for _ in extra], key=repr)
    assert sorted(calls, key=repr) == expected_calls
    columns = _columns(results)
    for name, k in names_to_k.items():
        assert columns[name] == [[k * 10]]
    if extra:
        assert columns[ROOT] == [[SENTINEL]]


def test_one_consumer_reading_both_contexts_raises() -> None:
    root = make_root_fg("TwinSplitBoth", __name__, ROOT, [1, 2], append_columns=True)

    class TwinSplitBothConsumerFeatureGroup(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"both"}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
            return {Feature(ROOT, Options(context={"k": 1})), Feature(ROOT, Options(context={"k": 2}))}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {"both": data[ROOT]}

    with pytest.raises(ValueError, match="from two steps"):
        mloda.run_all(
            ["both"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({root, TwinSplitBothConsumerFeatureGroup}),
        )


def test_two_direct_root_twins_with_differing_context_raise() -> None:
    root = make_root_fg("TwinSplitDirect", __name__, ROOT, [1, 2], append_columns=True)

    with pytest.raises(ValueError, match="reached twice"):
        mloda.run_all(
            [Feature(ROOT, Options(context={"k": 1})), Feature(ROOT, Options(context={"k": 2}))],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({root}),
        )


@pytest.mark.parametrize("framework", [PythonDictFramework, PyArrowTable], ids=["python_dict", "pyarrow"])
def test_second_level_consumers_follow_their_own_root_copy(framework: Any) -> None:
    calls: list[list[Any]] = []
    prefix = f"TwinSplitTwoLevel{framework.__name__}"
    root = _context_root(prefix, calls)
    consumers = _consumer_groups(prefix, {"c1": 1, "c2": 2}, one_group=False)
    sources = {"d1": "c1", "d2": "c2"}

    def feature_names_supported(cls: Any) -> set[str]:
        return set(sources)

    def input_features(self: Any, options: Options, feature_name: FeatureName) -> set[Feature]:
        return {Feature(sources[str(feature_name)])}

    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        out: dict[str, list[Any]] = {}
        for feature in features.features:
            column = data[sources[str(feature.name)]]
            out[str(feature.name)] = list(column) if isinstance(data, dict) else list(column.to_pylist())
        return out

    level_two = type(
        f"{prefix}LevelTwoFeatureGroup",
        (FeatureGroup,),
        {
            "__module__": __name__,
            "feature_names_supported": classmethod(feature_names_supported),
            "input_features": input_features,
            "calculate_feature": classmethod(calculate_feature),
        },
    )

    results = mloda.run_all(
        ["d1", "d2"],
        compute_frameworks=[framework],
        plugin_collector=PluginCollector.enabled_feature_groups({root, *consumers, level_two}),
    )

    assert sorted(calls, key=repr) == [[1], [2]]
    columns = _columns(results)
    assert columns["d1"] == [[10]]
    assert columns["d2"] == [[20]]


def test_two_level_chainer_pushing_ident_still_prepares() -> None:
    root = make_root_fg("TwinSplitTwoLevelChainer", __name__, "Sales", [1, 2])
    first = ChainedFeatureGroupTest.OPERATION_ID
    second = ChainedFeatureGroupTest_B.OPERATION_ID
    requests = [
        f"Sales__{first}identifier1",
        f"Sales__{first}identifier2",
        f"Sales__{first}identifier1__{second}identifier2",
        f"Sales__{first}identifier2__{second}identifier2",
    ]

    session = mloda.prepare(
        requests,
        compute_frameworks=[PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups(
            {root, ChainedFeatureGroupTest, ChainedFeatureGroupTest_B}
        ),
    )

    assert session.engine is not None


@pytest.mark.parametrize("with_plain", [False, True], ids=["chainers_only", "chainers_and_plain"])
def test_chainer_pushing_ident_onto_its_source_still_prepares(with_plain: bool) -> None:
    root = make_root_fg(f"TwinSplitChainer{with_plain}", __name__, "Sales", [1, 2])
    operation = ChainedContextFeatureGroupTest.OPERATION_ID
    requests = [f"Sales__{operation}identifier1", f"Sales__{operation}identifier2"]
    if with_plain:
        requests.append("Sales")

    session = mloda.prepare(
        requests,
        compute_frameworks=[PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups({root, ChainedContextFeatureGroupTest}),
    )

    assert session.engine is not None
