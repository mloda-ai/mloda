"""One compute framework wins every reduction, whatever the set iteration order.
Set iteration over class objects is id-based, so the reduction pins to the lowest class name.
"""

import importlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pytest

from mloda.provider import BaseInputData, DataCreator, FeatureSet
from mloda.user import PluginCollector, mloda

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_links import ResolveLinks
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.helpers.probe_runner import run_probes

_PROBE = Path(__file__).with_name("determinism_probe.py")
# Each probe is a fresh interpreter importing PyArrowTable, so the count is what the gate budget allows.
_PROBE_PROCESSES = 5
_PROBE_EXPECTED = {"feature": "PyArrowTable", "trekker_left": "PyArrowTable", "trekker_right": "PyArrowTable"}


class DeterminismLeftFeatureGroup(FeatureGroup):
    pass


class DeterminismRightFeatureGroup(FeatureGroup):
    pass


def _throwaway_frameworks() -> tuple[type[ComputeFramework], ...]:
    """Defined per call and unavailable, so discovery never sees them; Zz sorts behind every shipped name."""

    class ZzZuluThrowawayFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

    class ZzAlfaThrowawayFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

    class ZzTangoThrowawayFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

    class ZzBravoThrowawayFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

    # Definition order deliberately disagrees with name order.
    return (ZzZuluThrowawayFramework, ZzAlfaThrowawayFramework, ZzTangoThrowawayFramework, ZzBravoThrowawayFramework)


def _same_name_frameworks() -> tuple[type[ComputeFramework], type[ComputeFramework]]:
    """Two frameworks sharing a class name, separated only by qualname."""

    def alfa() -> type[ComputeFramework]:
        class ZzSharedNameThrowawayFramework(ComputeFramework):
            @staticmethod
            def is_available() -> bool:
                return False

        return ZzSharedNameThrowawayFramework

    def bravo() -> type[ComputeFramework]:
        class ZzSharedNameThrowawayFramework(ComputeFramework):
            @staticmethod
            def is_available() -> bool:
                return False

        return ZzSharedNameThrowawayFramework

    return alfa(), bravo()


def _link() -> Link:
    return Link.inner(
        JoinSpec(DeterminismLeftFeatureGroup, "idx"),
        JoinSpec(DeterminismRightFeatureGroup, "idx"),
    )


def test_select_deterministic_ignores_input_order() -> None:
    forward = ComputeFramework.select_deterministic([PandasDataFrame, PyArrowTable])
    backward = ComputeFramework.select_deterministic([PyArrowTable, PandasDataFrame])

    assert forward is backward
    assert forward is PandasDataFrame


def test_select_deterministic_returns_the_expected_name() -> None:
    zulu, alfa, tango, bravo = _throwaway_frameworks()
    expectations: list[tuple[tuple[type[ComputeFramework], ...], str]] = [
        ((zulu, alfa), "ZzAlfaThrowawayFramework"),
        ((zulu, tango), "ZzTangoThrowawayFramework"),
        ((tango, bravo), "ZzBravoThrowawayFramework"),
        ((zulu, tango, bravo), "ZzBravoThrowawayFramework"),
        ((zulu, alfa, tango, bravo), "ZzAlfaThrowawayFramework"),
    ]

    for group, expected in expectations:
        assert ComputeFramework.select_deterministic(set(group)).get_class_name() == expected


def test_select_deterministic_breaks_a_shared_class_name_by_qualname() -> None:
    """A shared class name would otherwise fall back to input order, which for a set is id-based."""
    alfa, bravo = _same_name_frameworks()

    forward = ComputeFramework.select_deterministic([alfa, bravo])
    backward = ComputeFramework.select_deterministic([bravo, alfa])

    assert forward is backward
    assert forward is alfa
    assert ".alfa." in forward.__qualname__, forward.__qualname__


def test_select_deterministic_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        ComputeFramework.select_deterministic([])


def test_the_throwaway_frameworks_stay_out_of_plugin_discovery() -> None:
    """get_cfw_subclasses is what planning consults; nothing this module defines may reach it."""
    held = set(_throwaway_frameworks()) | set(_same_name_frameworks())

    discovered = PreFilterPlugins.get_cfw_subclasses()

    assert not discovered & held, f"test-only frameworks reached discovery: {discovered & held}"
    leaked = sorted(cfw.get_class_name() for cfw in discovered if cfw.__module__ == __name__)
    assert leaked == [], f"test-only frameworks leaked into plugin discovery: {leaked}"


def test_link_trekker_key_reduces_both_sides_to_the_expected_framework() -> None:
    link = _link()

    key = ResolveLinks(Graph()).create_link_trekker_key(
        link, {PyArrowTable, PandasDataFrame}, {PyArrowTable, PandasDataFrame}
    )

    assert key == (link, PandasDataFrame, PandasDataFrame)


def test_link_trekker_key_reduces_every_framework_pair_the_same_way() -> None:
    resolver = ResolveLinks(Graph())
    zulu, alfa, tango, bravo = _throwaway_frameworks()
    expectations: list[tuple[tuple[type[ComputeFramework], type[ComputeFramework]], str]] = [
        ((zulu, alfa), "ZzAlfaThrowawayFramework"),
        ((zulu, tango), "ZzTangoThrowawayFramework"),
        ((tango, bravo), "ZzBravoThrowawayFramework"),
    ]

    for (left, right), expected in expectations:
        link = _link()
        key = resolver.create_link_trekker_key(link, {left, right}, {left, right})

        assert key[0] is link
        assert key[1].get_class_name() == expected
        assert key[2].get_class_name() == expected


def test_link_trekker_key_keeps_single_framework_sides() -> None:
    link = _link()

    key = ResolveLinks(Graph()).create_link_trekker_key(link, {PandasDataFrame}, {PyArrowTable})

    assert key == (link, PandasDataFrame, PyArrowTable)


def test_feature_compute_framework_is_the_expected_framework() -> None:
    feature = Feature("determinism_feature")
    feature.compute_frameworks = {PyArrowTable, PandasDataFrame}

    assert feature.get_compute_framework() is PandasDataFrame


def test_feature_compute_framework_is_the_expected_framework_for_every_pair() -> None:
    zulu, alfa, tango, bravo = _throwaway_frameworks()
    expectations: list[tuple[tuple[type[ComputeFramework], type[ComputeFramework]], str]] = [
        ((zulu, alfa), "ZzAlfaThrowawayFramework"),
        ((zulu, tango), "ZzTangoThrowawayFramework"),
        ((tango, bravo), "ZzBravoThrowawayFramework"),
    ]

    for (left, right), expected in expectations:
        feature = Feature("determinism_feature")
        feature.compute_frameworks = {left, right}

        assert feature.get_compute_framework().get_class_name() == expected


def test_feature_compute_framework_keeps_a_single_framework() -> None:
    feature = Feature("determinism_single_feature")
    feature.compute_frameworks = {PyArrowTable}

    assert feature.get_compute_framework() is PyArrowTable


def test_feature_compute_framework_rejects_an_empty_set() -> None:
    feature = Feature("determinism_empty_feature")
    feature.compute_frameworks = set()

    with pytest.raises(ValueError, match="determinism_empty_feature"):
        feature.get_compute_framework()


# Fresh interpreters cost roughly a second each, so this one needs more than the suite-wide per-test budget.
@pytest.mark.timeout(60)
def test_fresh_interpreters_reduce_to_the_same_frameworks() -> None:
    outputs = run_probes(_PROBE, _PROBE_PROCESSES)

    assert len(outputs) == _PROBE_PROCESSES, f"expected {_PROBE_PROCESSES} probe results, got {len(outputs)}"
    for position, output in enumerate(outputs):
        assert output == _PROBE_EXPECTED, f"probe {position} reduced to {output}, expected {_PROBE_EXPECTED}"


# ---------------------------------------------------------------------------
# Connection-aware default
# ---------------------------------------------------------------------------

_BASE = "mloda_plugins.compute_framework.base_implementations"
# (module, class name, expected requirement name); the modules guard their backend imports
_REQUIREMENTS: list[tuple[str, str, str]] = [
    (f"{_BASE}.pandas.dataframe", "PandasDataFrame", "NONE"),
    (f"{_BASE}.polars.dataframe", "PolarsDataFrame", "NONE"),
    (f"{_BASE}.polars.lazy_dataframe", "PolarsLazyDataFrame", "NONE"),
    (f"{_BASE}.pyarrow.table", "PyArrowTable", "NONE"),
    (f"{_BASE}.python_dict.python_dict_framework", "PythonDictFramework", "NONE"),
    (f"{_BASE}.spark.spark_framework", "SparkFramework", "SELF_MANAGED"),
    (f"{_BASE}.iceberg.iceberg_framework", "IcebergFramework", "SELF_MANAGED"),
    (f"{_BASE}.duckdb.duckdb_framework", "DuckDBFramework", "REQUIRED"),
    (f"{_BASE}.sqlite.sqlite_framework", "SqliteFramework", "REQUIRED"),
]


def _load_framework(module: str, name: str) -> type[ComputeFramework]:
    framework: type[ComputeFramework] = getattr(importlib.import_module(module), name)
    return framework


@pytest.mark.parametrize(("module", "name", "expected"), _REQUIREMENTS)
def test_connection_requirement_per_shipped_framework(module: str, name: str, expected: str) -> None:
    from mloda.core.abstract_plugins.components.connection_requirement import ConnectionRequirement

    framework = _load_framework(module, name)

    assert framework.connection_requirement() is ConnectionRequirement[expected]


def test_connection_requirement_members() -> None:
    from mloda.core.abstract_plugins.components.connection_requirement import ConnectionRequirement

    assert [member.name for member in ConnectionRequirement] == ["NONE", "SELF_MANAGED", "REQUIRED"]


_MODULE_OF = {name: module for module, name, _ in _REQUIREMENTS}
_RANK_CASES: list[tuple[list[str], str]] = [
    (["DuckDBFramework", "PyArrowTable"], "PyArrowTable"),
    (["PyArrowTable", "DuckDBFramework"], "PyArrowTable"),
    (["SqliteFramework", "DuckDBFramework"], "DuckDBFramework"),
    (["DuckDBFramework", "SqliteFramework"], "DuckDBFramework"),
    (["DuckDBFramework", "SparkFramework"], "SparkFramework"),
    (["SparkFramework", "PyArrowTable"], "PyArrowTable"),
]


@pytest.mark.parametrize(
    ("inputs", "expected"),
    _RANK_CASES,
    ids=[f"{'+'.join(i)}->{e}" for i, e in _RANK_CASES],
)
def test_select_deterministic_orders_by_connection_rank_then_name(inputs: list[str], expected: str) -> None:
    frameworks = [_load_framework(_MODULE_OF[name], name) for name in inputs]

    assert ComputeFramework.select_deterministic(frameworks).get_class_name() == expected


# --- planning and run ---------------------------------------------------------------------------


class _ConnAwareRoot(FeatureGroup):
    """Data-creator root named by NAME; subclasses differ only by class name and NAME."""

    NAME: ClassVar[str] = ""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({cls.NAME} if cls.NAME else set())

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({cls.NAME: [1, 2, 3]})


class ConnAwareDuckPyArrowRootFG(_ConnAwareRoot):
    NAME = "conn_aware_duck_pa_root"


class ConnAwareSqliteRootFG(_ConnAwareRoot):
    NAME = "conn_aware_sqlite_root"


class ConnAwareNoConnRootFG(_ConnAwareRoot):
    NAME = "conn_aware_noconn_root"


class ConnAwarePinnedRootFG(_ConnAwareRoot):
    NAME = "conn_aware_pinned_root"


@pytest.fixture
def sqlite_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


def _frameworks() -> set[type[ComputeFramework]]:
    pytest.importorskip("duckdb")
    duckdb_fw = _load_framework(f"{_BASE}.duckdb.duckdb_framework", "DuckDBFramework")
    sqlite_fw = _load_framework(f"{_BASE}.sqlite.sqlite_framework", "SqliteFramework")
    return {duckdb_fw, sqlite_fw}


def _compute_framework_names(
    feature: Feature | str, fg: type[FeatureGroup], frameworks: set[type[ComputeFramework]]
) -> list[str | None]:
    steps = mloda.explain(
        [feature], compute_frameworks=frameworks, plugin_collector=PluginCollector.enabled_feature_groups({fg})
    )
    return [step.compute_framework_name for step in steps if step.step_kind == "compute"]


def test_run_all_unrestricted_root_avoids_unconnected_duckdb() -> None:
    pytest.importorskip("duckdb")
    duckdb_fw = _load_framework(f"{_BASE}.duckdb.duckdb_framework", "DuckDBFramework")

    result = mloda.run_all(
        ["conn_aware_duck_pa_root"],
        compute_frameworks={duckdb_fw, PyArrowTable},
        plugin_collector=PluginCollector.enabled_feature_groups({ConnAwareDuckPyArrowRootFG}),
    )

    assert len(result) == 1
    assert isinstance(result[0], pa.Table)


def test_root_with_sqlite_connection_plans_on_sqlite(sqlite_conn: sqlite3.Connection) -> None:
    feature = Feature("conn_aware_sqlite_root", options={"ConnAwareSqliteRootFG": sqlite_conn})

    names = _compute_framework_names(feature, ConnAwareSqliteRootFG, _frameworks())

    assert names == ["SqliteFramework"]


def test_root_without_any_connection_keeps_duckdb() -> None:
    names = _compute_framework_names("conn_aware_noconn_root", ConnAwareNoConnRootFG, _frameworks())

    assert names == ["DuckDBFramework"]


def test_pinned_duckdb_feature_keeps_its_pin(sqlite_conn: sqlite3.Connection) -> None:
    feature = Feature(
        "conn_aware_pinned_root",
        options={"ConnAwarePinnedRootFG": sqlite_conn},
        compute_framework="DuckDBFramework",
    )

    names = _compute_framework_names(feature, ConnAwarePinnedRootFG, _frameworks())

    assert names == ["DuckDBFramework"]
