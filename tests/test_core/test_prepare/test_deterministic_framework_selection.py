"""One compute framework wins every reduction, whatever the set iteration order.
Set iteration over class objects is id-based, so the reduction ranks candidates first
and breaks remaining ties by class name.
"""

import copy
import importlib
import inspect
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pyarrow as pa
import pytest

from mloda.provider import BaseInputData, DataCreator, FeatureSet
from mloda.user import DataAccessCollection, GlobalFilter, Index, Options, PluginCollector, mloda

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework, framework_rank_key
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plan_info import PlanStep
from mloda.core.core.engine import Engine
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.prepare.accessible_plugins import EnvironmentPreconditionError, PreFilterPlugins
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.helpers.probe_runner import run_probes

_PROBE = Path(__file__).with_name("determinism_probe.py")
# Each probe is a fresh interpreter importing PyArrowTable, so the count is what the gate budget allows.
_PROBE_PROCESSES = 5
_PROBE_EXPECTED = {"feature": "PyArrowTable", "trekker_left": "PyArrowTable", "trekker_right": "PyArrowTable"}


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


def _same_name_and_qualname_frameworks() -> tuple[type[ComputeFramework], type[ComputeFramework]]:
    """Two frameworks identical in class name and qualname, separated only by module."""

    def make(module: str) -> type[ComputeFramework]:
        class ZzSharedModuleThrowawayFramework(ComputeFramework):
            @staticmethod
            def is_available() -> bool:
                return False

        ZzSharedModuleThrowawayFramework.__module__ = module
        return ZzSharedModuleThrowawayFramework

    return make("zz_module_b"), make("zz_module_a")


def _pick(
    candidates: Iterable[type[ComputeFramework]], positions: dict[type[ComputeFramework], int] | None = None
) -> type[ComputeFramework]:
    return min(candidates, key=framework_rank_key(positions or {}))


def test_rank_key_ignores_input_order() -> None:
    forward = _pick([PandasDataFrame, PyArrowTable])
    backward = _pick([PyArrowTable, PandasDataFrame])

    assert forward is backward
    assert forward is PandasDataFrame


def test_rank_key_returns_the_expected_name() -> None:
    zulu, alfa, tango, bravo = _throwaway_frameworks()
    expectations: list[tuple[tuple[type[ComputeFramework], ...], str]] = [
        ((zulu, alfa), "ZzAlfaThrowawayFramework"),
        ((zulu, tango), "ZzTangoThrowawayFramework"),
        ((tango, bravo), "ZzBravoThrowawayFramework"),
        ((zulu, tango, bravo), "ZzBravoThrowawayFramework"),
        ((zulu, alfa, tango, bravo), "ZzAlfaThrowawayFramework"),
    ]

    for group, expected in expectations:
        assert _pick(set(group)).get_class_name() == expected


def test_rank_key_breaks_a_shared_class_name_by_qualname() -> None:
    """A shared class name would otherwise fall back to input order, which for a set is id-based."""
    alfa, bravo = _same_name_frameworks()

    forward = _pick([alfa, bravo])
    backward = _pick([bravo, alfa])

    assert forward is backward
    assert forward is alfa
    assert ".alfa." in forward.__qualname__, forward.__qualname__


def test_rank_key_breaks_a_shared_name_and_qualname_by_module() -> None:
    module_b, module_a = _same_name_and_qualname_frameworks()
    assert module_a.__qualname__ == module_b.__qualname__

    for order in ([module_b, module_a], [module_a, module_b]):
        assert _pick(order) is module_a


def test_the_throwaway_frameworks_stay_out_of_plugin_discovery() -> None:
    """get_cfw_subclasses is what planning consults; nothing this module defines may reach it."""
    held = set(_throwaway_frameworks()) | set(_same_name_frameworks()) | set(_same_name_and_qualname_frameworks())

    discovered = PreFilterPlugins.get_cfw_subclasses()

    assert not discovered & held, f"test-only frameworks reached discovery: {discovered & held}"
    leaked = sorted(cfw.get_class_name() for cfw in discovered if cfw.__module__ == __name__)
    assert leaked == [], f"test-only frameworks leaked into plugin discovery: {leaked}"


def test_a_link_between_unrestricted_roots_joins_both_sides_on_the_expected_framework() -> None:
    steps = _plan(["lku_out"], _LKU_GROUPS, [PandasDataFrame, PyArrowTable], links=_lku_links())

    assert _compute_names(steps, LkuLeftRoot) == {"PandasDataFrame"}
    assert _compute_names(steps, LkuRightRoot) == {"PandasDataFrame"}
    assert [step.compute_framework_name for step in steps if step.step_kind == "join"] == ["PandasDataFrame"]


def test_throwaway_framework_pairs_reduce_to_the_expected_framework() -> None:
    zulu, alfa, tango, bravo = _throwaway_frameworks()
    expectations: list[tuple[tuple[type[ComputeFramework], type[ComputeFramework]], str]] = [
        ((zulu, alfa), "ZzAlfaThrowawayFramework"),
        ((zulu, tango), "ZzTangoThrowawayFramework"),
        ((tango, bravo), "ZzBravoThrowawayFramework"),
    ]

    for (left, right), expected in expectations:
        assert _pick({left, right}).get_class_name() == expected
        assert _pick([right, left], {}).get_class_name() == expected


def test_a_link_between_unrestricted_roots_follows_a_non_default_preference() -> None:
    steps = _plan(["lku_out"], _LKU_GROUPS, [PyArrowTable, PandasDataFrame], links=_lku_links())

    assert _compute_names(steps, LkuLeftRoot) == {"PyArrowTable"}
    assert _compute_names(steps, LkuRightRoot) == {"PyArrowTable"}
    assert [step.compute_framework_name for step in steps if step.step_kind == "join"] == ["PyArrowTable"]


def test_a_link_keeps_single_framework_sides_declared() -> None:
    steps = _plan(["lkm_out"], _LKM_GROUPS, [PandasDataFrame, PyArrowTable], links=_lkm_links())

    (join,) = [step for step in steps if step.step_kind == "join"]
    assert join.declared_left_framework_names == ("PandasDataFrame",)
    assert join.declared_right_framework_names == ("PyArrowTable",)


def test_feature_compute_framework_raises_on_an_unchosen_multi_framework_set() -> None:
    feature = Feature("determinism_feature")
    feature.compute_frameworks = {PyArrowTable, PandasDataFrame}

    with pytest.raises(ValueError):
        feature.get_compute_framework()


def test_feature_compute_framework_raises_on_every_unchosen_pair() -> None:
    zulu, alfa, tango, bravo = _throwaway_frameworks()

    for left, right in ((zulu, alfa), (zulu, tango), (tango, bravo)):
        feature = Feature("determinism_feature")
        feature.compute_frameworks = {left, right}

        with pytest.raises(ValueError):
            feature.get_compute_framework()


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
    from mloda.provider import ConnectionRequirement

    framework = _load_framework(module, name)

    assert framework.connection_requirement() is ConnectionRequirement[expected]


def test_connection_requirement_members() -> None:
    from mloda.provider import ConnectionRequirement

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
def test_rank_key_orders_by_connection_rank_then_name(inputs: list[str], expected: str) -> None:
    frameworks = [_load_framework(_MODULE_OF[name], name) for name in inputs]

    assert _pick(frameworks).get_class_name() == expected


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


def _frameworks() -> list[type[ComputeFramework]]:
    pytest.importorskip("duckdb")
    duckdb_fw = _load_framework(f"{_BASE}.duckdb.duckdb_framework", "DuckDBFramework")
    sqlite_fw = _load_framework(f"{_BASE}.sqlite.sqlite_framework", "SqliteFramework")
    return [duckdb_fw, sqlite_fw]


def _compute_framework_names(
    feature: Feature | str, fg: type[FeatureGroup], frameworks: list[type[ComputeFramework]]
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
        compute_frameworks=[PyArrowTable, duckdb_fw],
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


# --- run-level framework preference -------------------------------------------------------------


class PreferencePandasPyArrowRootFG(_ConnAwareRoot):
    NAME = "preference_pd_pa_root"


class PreferenceRequiredRootFG(_ConnAwareRoot):
    NAME = "preference_required_root"


class PreferenceUnavailableRootFG(_ConnAwareRoot):
    NAME = "preference_unavailable_root"


def test_preference_first_listed_among_candidates_wins() -> None:
    assert _pick([PandasDataFrame, PyArrowTable], {PyArrowTable: 0, PandasDataFrame: 1}) is PyArrowTable
    assert _pick([PyArrowTable, PandasDataFrame], {PandasDataFrame: 0, PyArrowTable: 1}) is PandasDataFrame


def test_preference_ranks_unlisted_candidates_after_listed_in_default_order() -> None:
    zulu, alfa, _, bravo = _throwaway_frameworks()

    assert _pick({alfa, zulu, bravo}, {zulu: 0}) is zulu
    assert _pick({alfa, bravo, PandasDataFrame}, {zulu: 0}) is PandasDataFrame


def test_unlisted_candidates_rank_after_every_listed_one_when_positions_repeat() -> None:
    zulu, alfa, _, bravo = _throwaway_frameworks()

    assert _pick({alfa, zulu, bravo}, {zulu: 1, bravo: 1}) is bravo
    assert _pick({alfa, zulu, PandasDataFrame}, {zulu: 1, bravo: 1}) is zulu


def test_a_shared_position_is_broken_by_the_module() -> None:
    module_b, module_a = _same_name_and_qualname_frameworks()
    positions: dict[type[ComputeFramework], int] = {module_b: 0, module_a: 0, PandasDataFrame: 1}

    assert _pick({module_b, module_a, PandasDataFrame}, positions) is module_a


def test_no_preference_leaves_the_default_unchanged() -> None:
    assert _pick([PyArrowTable, PandasDataFrame], {}) is PandasDataFrame


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        (["PyArrowTable", "PandasDataFrame"], ["PyArrowTable"]),
        (["PandasDataFrame", "PyArrowTable"], ["PandasDataFrame"]),
    ],
)
def test_planning_follows_the_listed_order(order: list[Any], expected: list[str | None]) -> None:
    frameworks = [_load_framework(_MODULE_OF[name], name) for name in order]

    names = _compute_framework_names("preference_pd_pa_root", PreferencePandasPyArrowRootFG, frameworks)

    assert names == expected


def test_required_framework_listed_first_without_connection_still_loses() -> None:
    sqlite_fw = _load_framework(_MODULE_OF["SqliteFramework"], "SqliteFramework")

    names = _compute_framework_names("preference_required_root", PreferenceRequiredRootFG, [sqlite_fw, PandasDataFrame])

    assert names == ["PandasDataFrame"]


def test_unavailable_framework_listed_first_falls_back_to_the_next_listed() -> None:
    unavailable = _throwaway_frameworks()[0]

    names = _compute_framework_names(
        "preference_unavailable_root", PreferenceUnavailableRootFG, [unavailable, PyArrowTable, PandasDataFrame]
    )

    assert names == ["PyArrowTable"]


class PreferencePinnedPandasRootFG(_ConnAwareRoot):
    NAME = "preference_pinned_pandas_root"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({cls.NAME: [1, 2, 3]})


def test_feature_pinned_to_pandas_still_runs_on_pandas_under_a_pyarrow_first_run_list() -> None:
    feature = Feature("preference_pinned_pandas_root", compute_framework="PandasDataFrame")

    result = mloda.run_all(
        [feature],
        compute_frameworks=["PyArrowTable", "PandasDataFrame"],
        plugin_collector=PluginCollector.enabled_feature_groups({PreferencePinnedPandasRootFG}),
    )

    assert len(result) == 1
    assert isinstance(result[0], pd.DataFrame)


# --- central choice: Feature.chosen_compute_framework and explicit positions ----------------------


def _two_framework_feature(name: str = "chosen_feature") -> Feature:
    feature = Feature(name)
    feature.compute_frameworks = {PyArrowTable, PandasDataFrame}
    return feature


def test_chosen_compute_framework_defaults_to_none() -> None:
    assert _two_framework_feature().chosen_compute_framework is None


def test_get_compute_framework_returns_the_chosen_one() -> None:
    feature = _two_framework_feature()
    feature.chosen_compute_framework = PyArrowTable

    assert feature.get_compute_framework() is PyArrowTable


def test_chosen_compute_framework_is_not_part_of_identity() -> None:
    plain = _two_framework_feature()
    chosen = _two_framework_feature()
    chosen.chosen_compute_framework = PyArrowTable

    assert plain == chosen
    assert hash(plain) == hash(chosen)
    assert plain.similarity_hash(frozenset()) == chosen.similarity_hash(frozenset())
    assert plain.base_similarity_hash(frozenset()) == chosen.base_similarity_hash(frozenset())


def test_copy_keeps_the_chosen_compute_framework() -> None:
    feature = _two_framework_feature()
    feature.chosen_compute_framework = PyArrowTable

    assert copy.copy(feature).chosen_compute_framework is PyArrowTable


@pytest.mark.parametrize(
    ("positions", "candidates", "expected"),
    [
        ({PyArrowTable: 0, PandasDataFrame: 1}, [PandasDataFrame, PyArrowTable], PyArrowTable),
        ({PandasDataFrame: 0, PyArrowTable: 1}, [PyArrowTable, PandasDataFrame], PandasDataFrame),
        ({}, [PyArrowTable, PandasDataFrame], PandasDataFrame),
        (None, [PyArrowTable, PandasDataFrame], PandasDataFrame),
        ({PyArrowTable: 0}, [PandasDataFrame, PyArrowTable], PyArrowTable),
    ],
    ids=["pa_first", "pd_first", "empty", "none", "unlisted_after_listed"],
)
def test_rank_key_follows_explicit_positions(
    positions: dict[type[ComputeFramework], int] | None,
    candidates: list[type[ComputeFramework]],
    expected: type[ComputeFramework],
) -> None:
    assert _pick(candidates, positions) is expected


def test_rank_key_positions_tie_breaks_by_module_after_name() -> None:
    module_b, module_a = _same_name_and_qualname_frameworks()

    assert _pick({module_b, module_a}, {module_b: 0, module_a: 0}) is module_a


def test_rank_key_positions_rank_connection_before_name() -> None:
    sqlite_fw = _load_framework(_MODULE_OF["SqliteFramework"], "SqliteFramework")

    assert _pick([sqlite_fw, PyArrowTable], {sqlite_fw: 0, PyArrowTable: 0}) is PyArrowTable


def test_framework_rank_key_orders_by_position_then_default() -> None:
    zulu, alfa, _, bravo = _throwaway_frameworks()
    key = framework_rank_key({zulu: 0, bravo: 1})

    candidates: list[type[ComputeFramework]] = [alfa, bravo, zulu, PandasDataFrame]

    assert sorted(candidates, key=key) == [zulu, bravo, PandasDataFrame, alfa]


def test_framework_rank_key_without_positions_is_the_default_order() -> None:
    key = framework_rank_key({})

    candidates: list[type[ComputeFramework]] = [PyArrowTable, PandasDataFrame]

    assert sorted(candidates, key=key) == [PandasDataFrame, PyArrowTable]


# --- the ContextVar is gone; the engine takes the preference ---------------------------------------


def test_the_preference_contextvar_no_longer_exists() -> None:
    module = importlib.import_module("mloda.core.abstract_plugins.compute_framework")

    assert not hasattr(module, "framework_preference")
    assert not hasattr(module, "_framework_position")


def test_engine_takes_a_framework_preference_keyword() -> None:
    assert "framework_preference" in inspect.signature(Engine.__init__).parameters


# --- central choice wiring: plans through the public API ------------------------------------------


class _PlanRoot(FeatureGroup):
    """Data-creator root; FW_NAME restricts it, None leaves it unrestricted."""

    NAMES: ClassVar[tuple[str, ...]] = ()
    FW_NAME: ClassVar[str | None] = None
    INDEXES: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(set(cls.NAMES))

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({name: [1, 2, 3] for name in cls.NAMES})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None if cls.FW_NAME is None else {_load_framework(_MODULE_OF[cls.FW_NAME], cls.FW_NAME)}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index((name,)) for name in cls.INDEXES] or None


class _PlanConsumer(FeatureGroup):
    """Consumer of INPUTS producing OUTPUT; FW_NAME restricts it, None leaves it unrestricted."""

    INPUTS: ClassVar[tuple[str, ...]] = ()
    OUTPUT: ClassVar[str] = ""
    FW_NAME: ClassVar[str | None] = None

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name) for name in self.INPUTS}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None if cls.FW_NAME is None else {_load_framework(_MODULE_OF[cls.FW_NAME], cls.FW_NAME)}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.OUTPUT} if cls.OUTPUT else set()


def _target(name: str) -> type[ComputeFramework]:
    if name.startswith("Polars"):
        pytest.importorskip("polars")
    return _load_framework(_MODULE_OF[name], name)


_TARGETS = [
    pytest.param("PyArrowTable", id="pyarrow"),
    pytest.param("PolarsDataFrame", id="polars"),
    pytest.param("PolarsLazyDataFrame", id="polars_lazy"),
]


def _plan(
    features: list[Feature | str],
    groups: set[type[FeatureGroup]],
    frameworks: list[type[ComputeFramework]],
    links: set[Link] | None = None,
) -> list[PlanStep]:
    return mloda.explain(
        features,
        compute_frameworks=frameworks,
        links=links,
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
    )


def _compute_steps(steps: list[PlanStep], group: type[FeatureGroup]) -> list[PlanStep]:
    return [step for step in steps if step.step_kind == "compute" and step.feature_group is group]


def _compute_names(steps: list[PlanStep], group: type[FeatureGroup]) -> set[str | None]:
    return {step.compute_framework_name for step in _compute_steps(steps, group)}


def _transforms(steps: list[PlanStep]) -> list[PlanStep]:
    return [step for step in steps if step.step_kind == "transform"]


# P1: an unrestricted root feeds a consumer restricted to the target.
class P1Root(_PlanRoot):
    NAMES = ("p1_root",)


class P1ConsumerPyArrowTable(_PlanConsumer):
    INPUTS = ("p1_root",)
    OUTPUT = "p1_out"
    FW_NAME = "PyArrowTable"


class P1ConsumerPolarsDataFrame(P1ConsumerPyArrowTable):
    FW_NAME = "PolarsDataFrame"


class P1ConsumerPolarsLazyDataFrame(P1ConsumerPyArrowTable):
    FW_NAME = "PolarsLazyDataFrame"


# P3: a source restricted to the target feeds an unrestricted consumer.
class P3SourcePyArrowTable(_PlanRoot):
    NAMES = ("p3_src",)
    FW_NAME = "PyArrowTable"


class P3SourcePolarsDataFrame(P3SourcePyArrowTable):
    FW_NAME = "PolarsDataFrame"


class P3SourcePolarsLazyDataFrame(P3SourcePyArrowTable):
    FW_NAME = "PolarsLazyDataFrame"


class P3Consumer(_PlanConsumer):
    INPUTS = ("p3_src",)
    OUTPUT = "p3_out"


# P2: a link joins two unrestricted roots; the link child is restricted to the target.
class P2RootA(_PlanRoot):
    NAMES = ("p2_a",)


class P2RootB(_PlanRoot):
    NAMES = ("p2_b",)


class P2ChildPyArrowTable(_PlanConsumer):
    INPUTS = ("p2_a", "p2_b")
    OUTPUT = "p2_out"
    FW_NAME = "PyArrowTable"


class P2ChildPolarsDataFrame(P2ChildPyArrowTable):
    FW_NAME = "PolarsDataFrame"


class P2ChildPolarsLazyDataFrame(P2ChildPyArrowTable):
    FW_NAME = "PolarsLazyDataFrame"


_P1_CONSUMERS: dict[str, type[FeatureGroup]] = {
    "PyArrowTable": P1ConsumerPyArrowTable,
    "PolarsDataFrame": P1ConsumerPolarsDataFrame,
    "PolarsLazyDataFrame": P1ConsumerPolarsLazyDataFrame,
}
_P3_SOURCES: dict[str, type[FeatureGroup]] = {
    "PyArrowTable": P3SourcePyArrowTable,
    "PolarsDataFrame": P3SourcePolarsDataFrame,
    "PolarsLazyDataFrame": P3SourcePolarsLazyDataFrame,
}
_P2_CHILDREN: dict[str, type[FeatureGroup]] = {
    "PyArrowTable": P2ChildPyArrowTable,
    "PolarsDataFrame": P2ChildPolarsDataFrame,
    "PolarsLazyDataFrame": P2ChildPolarsLazyDataFrame,
}


@pytest.mark.parametrize("target", _TARGETS)
def test_p1_unrestricted_root_moves_onto_the_restricted_consumers_framework(target: str) -> None:
    framework = _target(target)

    steps = _plan(["p1_out"], {P1Root, _P1_CONSUMERS[target]}, [PandasDataFrame, framework])

    assert _compute_names(steps, P1Root) == {target}
    assert _transforms(steps) == []


@pytest.mark.parametrize("target", _TARGETS)
def test_p3_unrestricted_consumer_moves_onto_the_restricted_sources_framework(target: str) -> None:
    framework = _target(target)

    steps = _plan(["p3_out"], {P3Consumer, _P3_SOURCES[target]}, [PandasDataFrame, framework])

    assert _compute_names(steps, P3Consumer) == {target}
    assert _transforms(steps) == []


@pytest.mark.parametrize("target", _TARGETS)
def test_p2_link_between_unrestricted_roots_joins_on_the_restricted_childs_framework(target: str) -> None:
    framework = _target(target)
    links = {Link.inner(JoinSpec(P2RootA, "p2_idx"), JoinSpec(P2RootB, "p2_idx"))}

    steps = _plan(["p2_out"], {P2RootA, P2RootB, _P2_CHILDREN[target]}, [PandasDataFrame, framework], links=links)

    assert [step.compute_framework_name for step in steps if step.step_kind == "join"] == [target]
    assert _compute_names(steps, P2RootA) == {target}
    assert _compute_names(steps, P2RootB) == {target}
    assert _transforms(steps) == []


# P4: two unlinked roots on one framework feed a consumer on another; nothing joins them.
class P4RootA(_PlanRoot):
    NAMES = ("p4_root_a",)
    FW_NAME = "PandasDataFrame"


class P4RootB(_PlanRoot):
    NAMES = ("p4_root_b",)
    FW_NAME = "PandasDataFrame"


class P4Consumer(_PlanConsumer):
    INPUTS = ("p4_root_a", "p4_root_b")
    OUTPUT = "p4_out"
    FW_NAME = "PyArrowTable"


def test_p4_unlinked_parents_raise_the_missing_links_error() -> None:
    with pytest.raises(ValueError, match="Link") as raised:
        _plan(["p4_out"], {P4RootA, P4RootB, P4Consumer}, [PandasDataFrame, PyArrowTable], links=set())

    assert "P4RootA" in str(raised.value)
    assert "P4RootB" in str(raised.value)


# A feature both requested and consumed keeps one read.
class RcRoot(_PlanRoot):
    NAMES = ("rc_root",)


class RcConsumer(_PlanConsumer):
    INPUTS = ("rc_root",)
    OUTPUT = "rc_out"
    FW_NAME = "PyArrowTable"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"rc_out": data["rc_root"]})


def test_a_feature_requested_and_consumed_has_one_step_on_the_consumers_framework() -> None:
    steps = _plan(["rc_root", "rc_out"], {RcRoot, RcConsumer}, [PandasDataFrame, PyArrowTable])

    assert len(_compute_steps(steps, RcRoot)) == 1
    assert _compute_names(steps, RcRoot) == {"PyArrowTable"}
    assert _transforms(steps) == []


_RC_GROUPS: set[type[FeatureGroup]] = {RcRoot, RcConsumer}
_RC_FRAMEWORKS: list[type[ComputeFramework]] = [PandasDataFrame, PyArrowTable]
_RC_REQUESTS = [pytest.param(["rc_root"], id="alone"), pytest.param(["rc_root", "rc_out"], id="together")]


def _run_rc(features: list[Feature | str], output_framework: str | type[ComputeFramework] | None = None) -> Any:
    return mloda.run_all(
        features,
        compute_frameworks=_RC_FRAMEWORKS,
        plugin_collector=PluginCollector.enabled_feature_groups(_RC_GROUPS),
        output_framework=output_framework,
    )


@pytest.mark.parametrize("output", ["PandasDataFrame", PandasDataFrame], ids=["name", "class"])
@pytest.mark.parametrize("features", _RC_REQUESTS)
def test_output_framework_returns_every_frame_in_that_framework(output: Any, features: list[Feature | str]) -> None:
    result = _run_rc(features, output)

    assert len(result) == len(features)
    assert all(isinstance(frame, pd.DataFrame) for frame in result)
    for step, frame in result.frames():
        assert step.result_framework is PandasDataFrame
        assert type(frame) is step.result_framework.expected_data_framework()


@pytest.mark.parametrize(
    ("features", "expected_names"),
    [(["rc_root"], None), (["rc_root", "rc_out"], {"PyArrowTable"})],
    ids=["alone", "together"],
)
def test_result_framework_names_the_type_each_frame_comes_back_in(
    features: list[Feature | str], expected_names: set[str] | None
) -> None:
    result = _run_rc(features)

    for step, frame in result.frames():
        assert step.result_framework is not None
        assert type(frame) is step.result_framework.expected_data_framework()
    if expected_names is not None:
        assert {step.result_framework_name for step, _ in result.frames()} == expected_names


def test_stream_all_honors_output_framework() -> None:
    stream = mloda.stream_all(
        ["rc_root", "rc_out"],
        compute_frameworks=_RC_FRAMEWORKS,
        plugin_collector=PluginCollector.enabled_feature_groups(_RC_GROUPS),
        output_framework="PandasDataFrame",
    )

    frames = list(stream)

    assert len(frames) == 2
    assert all(isinstance(frame, pd.DataFrame) for frame in frames)


def test_a_required_connection_output_framework_without_a_connection_raises_at_prepare() -> None:
    sqlite_fw = _load_framework(_MODULE_OF["SqliteFramework"], "SqliteFramework")

    with pytest.raises(ValueError, match="SqliteFramework"):
        mloda.prepare(
            ["rc_root"],
            compute_frameworks=_RC_FRAMEWORKS,
            plugin_collector=PluginCollector.enabled_feature_groups(_RC_GROUPS),
            output_framework=sqlite_fw,
        )


def test_a_required_connection_output_framework_with_a_connection_prepares(sqlite_conn: sqlite3.Connection) -> None:
    sqlite_fw = _load_framework(_MODULE_OF["SqliteFramework"], "SqliteFramework")
    access = DataAccessCollection(connections={sqlite_conn})

    session = mloda.prepare(
        ["rc_root"],
        compute_frameworks=_RC_FRAMEWORKS,
        data_access_collection=access,
        plugin_collector=PluginCollector.enabled_feature_groups(_RC_GROUPS),
        output_framework=sqlite_fw,
    )

    assert session.engine is not None
    assert session.engine.output_connection is sqlite_conn


class TfsDbRoot(_PlanRoot):
    NAMES = ("tfsdb_root",)
    FW_NAME = "PyArrowTable"


class TfsSqliteConsumer(_PlanConsumer):
    INPUTS = ("tfsdb_root",)
    OUTPUT = "tfsdb_sqlite_out"
    FW_NAME = "SqliteFramework"


class TfsDuckDBConsumer(_PlanConsumer):
    INPUTS = ("tfsdb_root",)
    OUTPUT = "tfsdb_duckdb_out"
    FW_NAME = "DuckDBFramework"


_TFS_DB_CASES = [
    pytest.param("SqliteFramework", TfsSqliteConsumer, "tfsdb_sqlite_out", id="sqlite"),
    pytest.param("DuckDBFramework", TfsDuckDBConsumer, "tfsdb_duckdb_out", id="duckdb"),
]
_TFS_DAC = pytest.mark.parametrize("with_dac", [False, True], ids=["no_dac", "dac_without_connection"])


def _tfs_request(fw_name: str, consumer: type[FeatureGroup], with_dac: bool, tmp_path: Path) -> dict[str, Any]:
    if fw_name == "DuckDBFramework":
        pytest.importorskip("duckdb")
    return {
        "compute_frameworks": [PyArrowTable, _load_framework(_MODULE_OF[fw_name], fw_name)],
        "plugin_collector": PluginCollector.enabled_feature_groups({TfsDbRoot, consumer}),
        "data_access_collection": DataAccessCollection(folders={str(tmp_path)}) if with_dac else None,
    }


@_TFS_DAC
@pytest.mark.parametrize(("fw_name", "consumer", "feature"), _TFS_DB_CASES)
@pytest.mark.parametrize("entry", ["prepare", "run_all"])
def test_a_transform_into_a_required_connection_framework_without_a_connection_raises_at_planning(
    entry: str, fw_name: str, consumer: type[FeatureGroup], feature: str, with_dac: bool, tmp_path: Path
) -> None:
    kwargs = _tfs_request(fw_name, consumer, with_dac, tmp_path)

    with pytest.raises(EnvironmentPreconditionError, match=fw_name) as raised:
        getattr(mloda, entry)([feature], **kwargs)

    assert "DataAccessCollection" in str(raised.value)


@_TFS_DAC
@pytest.mark.parametrize(("fw_name", "consumer", "feature"), _TFS_DB_CASES)
def test_diagnose_reports_a_transform_into_a_required_connection_framework_without_a_connection(
    fw_name: str, consumer: type[FeatureGroup], feature: str, with_dac: bool, tmp_path: Path
) -> None:
    kwargs = _tfs_request(fw_name, consumer, with_dac, tmp_path)

    diagnosis = mloda.diagnose([feature], **kwargs)

    assert diagnosis.complete is False
    assert diagnosis.message is not None
    assert fw_name in diagnosis.message
    assert "DataAccessCollection" in diagnosis.message


def test_a_same_type_hop_into_a_required_connection_framework_needs_no_connection() -> None:
    sqlite_cls = _load_framework(_MODULE_OF["SqliteFramework"], "SqliteFramework")
    to_cls = sqlite_cls
    tfs = type("_Tfs", (), {"from_framework": sqlite_cls, "to_framework": to_cls})()
    engine = Engine.__new__(Engine)
    engine.execution_planner = type("_Planner", (), {"tfs_collection": {"k": tfs}})()
    engine.data_access_collection = None

    assert engine._resolve_tfs_connection_map() == {}


def test_an_output_framework_outside_the_run_list_still_converts() -> None:
    expected = PythonDictFramework.expected_data_framework()

    result = _run_rc(["rc_root"], "PythonDictFramework")

    assert len(result) == 1
    assert all(type(frame) is expected for frame in result)


# An unrelated consumer must not move a root.
class UrRoot(_PlanRoot):
    NAMES = ("ur_root",)


class UrConsumer(_PlanConsumer):
    INPUTS = ("ur_root",)
    OUTPUT = "ur_out"


class UrOtherSource(_PlanRoot):
    NAMES = ("ur_other_src",)
    FW_NAME = "PyArrowTable"


class UrOtherConsumer(_PlanConsumer):
    INPUTS = ("ur_other_src",)
    OUTPUT = "ur_other_out"
    FW_NAME = "PyArrowTable"


def test_an_unrelated_consumer_does_not_change_the_roots_framework() -> None:
    groups: set[type[FeatureGroup]] = {UrRoot, UrConsumer, UrOtherSource, UrOtherConsumer}
    frameworks: list[type[ComputeFramework]] = [PandasDataFrame, PyArrowTable]

    alone = _plan(["ur_out"], groups, frameworks)
    together = _plan(["ur_out", "ur_other_out"], groups, frameworks)

    assert _compute_names(alone, UrRoot) == {"PandasDataFrame"}
    assert _compute_names(together, UrRoot) == _compute_names(alone, UrRoot)


# Index feature requested explicitly on the host's own feature group.
class IxLeft(_PlanRoot):
    NAMES = ("ix_left_val", "ix_lidx")
    INDEXES = ("ix_lidx",)


class IxRight(_PlanRoot):
    NAMES = ("ix_right_val", "ix_ridx")
    INDEXES = ("ix_ridx",)


class IxConsumer(_PlanConsumer):
    INPUTS = ("ix_left_val", "ix_right_val")
    OUTPUT = "ix_out"
    FW_NAME = "PyArrowTable"


_IX_GROUPS: set[type[FeatureGroup]] = {IxLeft, IxRight, IxConsumer}
_IX_FEATURES: list[Feature | str] = ["ix_out", "ix_lidx", "ix_ridx"]


def _ix_links() -> set[Link]:
    return {Link.inner(JoinSpec(IxLeft, "ix_lidx"), JoinSpec(IxRight, "ix_ridx"))}


def test_an_index_feature_requested_explicitly_shares_the_hosts_single_read() -> None:
    steps = _plan(_IX_FEATURES, _IX_GROUPS, [PandasDataFrame, PyArrowTable], links=_ix_links())

    for group in (IxLeft, IxRight):
        assert len(_compute_steps(steps, group)) == 1
        assert _compute_names(steps, group) == {"PyArrowTable"}
    assert _transforms(steps) == []


def test_every_feature_group_step_holds_one_chosen_framework() -> None:
    session = mloda.prepare(
        _IX_FEATURES,
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        links=_ix_links(),
        plugin_collector=PluginCollector.enabled_feature_groups(_IX_GROUPS),
    )
    assert session.engine is not None

    steps = [step for step in session.engine.execution_planner if isinstance(step, FeatureGroupStep)]

    assert len(steps) >= 3
    for step in steps:
        chosen = {feature.chosen_compute_framework for feature in step.features.features}
        assert chosen == {step.compute_framework}, f"{step.feature_group.__name__}: {chosen}"


# Link scenarios for the rewritten trekker-key intent tests.
class LkuLeftRoot(_PlanRoot):
    NAMES = ("lku_left",)


class LkuRightRoot(_PlanRoot):
    NAMES = ("lku_right",)


class LkuChild(_PlanConsumer):
    INPUTS = ("lku_left", "lku_right")
    OUTPUT = "lku_out"


class LkmLeftRoot(_PlanRoot):
    NAMES = ("lkm_left",)
    FW_NAME = "PandasDataFrame"


class LkmRightRoot(_PlanRoot):
    NAMES = ("lkm_right",)
    FW_NAME = "PyArrowTable"


class LkmChild(_PlanConsumer):
    INPUTS = ("lkm_left", "lkm_right")
    OUTPUT = "lkm_out"


_LKU_GROUPS: set[type[FeatureGroup]] = {LkuLeftRoot, LkuRightRoot, LkuChild}
_LKM_GROUPS: set[type[FeatureGroup]] = {LkmLeftRoot, LkmRightRoot, LkmChild}


def _lku_links() -> set[Link]:
    return {Link.inner(JoinSpec(LkuLeftRoot, "lku_idx"), JoinSpec(LkuRightRoot, "lku_idx"))}


def _lkm_links() -> set[Link]:
    return {Link.inner(JoinSpec(LkmLeftRoot, "lkm_idx"), JoinSpec(LkmRightRoot, "lkm_idx"))}


# --- a pinned and an unpinned request of one feature share one read ------------------------------


class PinnedUnpinnedSharedRootFG(_ConnAwareRoot):
    NAME = "pinned_unpinned_shared_root"


_SHARED_PLUGINS = PluginCollector.enabled_feature_groups({PinnedUnpinnedSharedRootFG})


def _shared_root_steps(
    features: list[Feature | str],
    plugins: PluginCollector = _SHARED_PLUGINS,
    global_filter: GlobalFilter | None = None,
    links: set[Link] | None = None,
) -> tuple[Any, list[FeatureGroupStep]]:
    session = mloda.prepare(
        features,
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        plugin_collector=plugins,
        global_filter=global_filter,
        links=links,
    )
    assert session.engine is not None
    steps = [
        step
        for step in session.engine.execution_planner
        if isinstance(step, FeatureGroupStep) and step.feature_group is PinnedUnpinnedSharedRootFG
    ]
    return session, steps


@pytest.mark.parametrize("pinned_first", [True, False], ids=["pinned_first", "unpinned_first"])
def test_pinned_and_unpinned_request_share_one_step_in_the_pinned_framework(pinned_first: bool) -> None:
    pinned = Feature(PinnedUnpinnedSharedRootFG.NAME, compute_framework="PyArrowTable")
    features: list[Feature | str] = [pinned, PinnedUnpinnedSharedRootFG.NAME]
    if not pinned_first:
        features.reverse()

    _, steps = _shared_root_steps(features)

    assert [step.compute_framework for step in steps] == [PyArrowTable]
    assert [f.chosen_compute_framework_reason for f in steps[0].features.features] == ["pinned"]

    result = mloda.run_all(
        features,
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        plugin_collector=_SHARED_PLUGINS,
    )

    assert len(result) == 1
    assert isinstance(result[0], pa.Table)
    assert result[0].column_names == [PinnedUnpinnedSharedRootFG.NAME]
    assert result[0][PinnedUnpinnedSharedRootFG.NAME].to_pylist() == [1, 2, 3]


def _shared_pair(pinned_first: bool) -> list[Feature | str]:
    features: list[Feature | str] = [
        Feature(PinnedUnpinnedSharedRootFG.NAME, compute_framework="PyArrowTable"),
        PinnedUnpinnedSharedRootFG.NAME,
    ]
    return features if pinned_first else features[::-1]


@pytest.mark.parametrize("pinned_first", [True, False], ids=["pinned_first", "unpinned_first"])
def test_pinned_and_unpinned_request_share_one_filtered_step(pinned_first: bool) -> None:
    global_filter = GlobalFilter()
    global_filter.add_filter(PinnedUnpinnedSharedRootFG.NAME, "equal", {"value": 2})
    features = _shared_pair(pinned_first)

    _, steps = _shared_root_steps(features, global_filter=global_filter)

    assert [step.compute_framework for step in steps] == [PyArrowTable]

    result = mloda.run_all(
        features,
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        plugin_collector=_SHARED_PLUGINS,
        global_filter=global_filter,
    )

    assert len(result) == 1
    assert result[0][PinnedUnpinnedSharedRootFG.NAME].to_pylist() == [2]


class PinnedUnpinnedSharedOtherRootFG(_ConnAwareRoot):
    NAME = "pinned_unpinned_shared_other"


_SHARED_LINK_PLUGINS = PluginCollector.enabled_feature_groups(
    {PinnedUnpinnedSharedRootFG, PinnedUnpinnedSharedOtherRootFG}
)


@pytest.mark.parametrize("pinned_first", [True, False], ids=["pinned_first", "unpinned_first"])
def test_pinned_and_unpinned_request_share_one_step_with_a_link(pinned_first: bool) -> None:
    links = {Link.inner(JoinSpec(PinnedUnpinnedSharedRootFG, "idx"), JoinSpec(PinnedUnpinnedSharedOtherRootFG, "idx"))}
    features = [*_shared_pair(pinned_first), PinnedUnpinnedSharedOtherRootFG.NAME]

    _, steps = _shared_root_steps(features, plugins=_SHARED_LINK_PLUGINS, links=links)

    assert [step.compute_framework for step in steps] == [PyArrowTable]


def _filter_pinned_to_pandas() -> GlobalFilter:
    global_filter = GlobalFilter()
    pandas_pin = Feature(PinnedUnpinnedSharedRootFG.NAME, compute_framework="PandasDataFrame")
    global_filter.add_filter(pandas_pin, "equal", {"value": 2})
    return global_filter


def test_a_filter_pinned_to_another_framework_plans_alike_in_both_orders() -> None:
    shapes = []
    for pinned_first in (True, False):
        _, steps = _shared_root_steps(_shared_pair(pinned_first), global_filter=_filter_pinned_to_pandas())
        shapes.append((len(steps), {step.compute_framework for step in steps}))

    assert shapes[0] == shapes[1]


_UNPINNED_SLOT = "unpinned"
_PIN_ORDERS = [
    pytest.param(["pa", "pd", _UNPINNED_SLOT], id="pins_first"),
    pytest.param([_UNPINNED_SLOT, "pa", "pd"], id="unpinned_first"),
    pytest.param(["pa", _UNPINNED_SLOT, "pd"], id="unpinned_middle"),
]


@pytest.mark.parametrize("order", _PIN_ORDERS)
def test_two_differing_pins_plus_unpinned_merge_nothing_in_any_order(order: list[str]) -> None:
    by_slot: dict[str, Feature | str] = {
        "pa": Feature(PinnedUnpinnedSharedRootFG.NAME, compute_framework="PyArrowTable"),
        "pd": Feature(PinnedUnpinnedSharedRootFG.NAME, compute_framework="PandasDataFrame"),
        _UNPINNED_SLOT: PinnedUnpinnedSharedRootFG.NAME,
    }

    _, steps = _shared_root_steps([by_slot[slot] for slot in order])

    assert len(steps) == 3


class PinnedUnpinnedSharedConsumerFG(_PlanConsumer):
    OUTPUT = "pinned_unpinned_shared_consumer"
    FW_NAME = "PyArrowTable"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({cls.OUTPUT: [1, 2, 3]})

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(PinnedUnpinnedSharedRootFG.NAME, compute_framework="PyArrowTable"),
            Feature(PinnedUnpinnedSharedRootFG.NAME),
        }


_SHARED_CONSUMER_PLUGINS = PluginCollector.enabled_feature_groups(
    {PinnedUnpinnedSharedRootFG, PinnedUnpinnedSharedConsumerFG}
)


def test_pinned_and_unpinned_consumer_inputs_share_one_read() -> None:
    out = PinnedUnpinnedSharedConsumerFG.OUTPUT

    _, steps = _shared_root_steps([out], plugins=_SHARED_CONSUMER_PLUGINS)

    assert [step.compute_framework for step in steps] == [PyArrowTable]

    result = mloda.run_all(
        [out], compute_frameworks=[PandasDataFrame, PyArrowTable], plugin_collector=_SHARED_CONSUMER_PLUGINS
    )

    assert len(result) == 1
    assert isinstance(result[0], pa.Table)
    assert result[0].column_names == [out]
    assert result[0][out].to_pylist() == [1, 2, 3]


def test_pinned_and_unpinned_request_with_differing_options_stay_two_steps() -> None:
    pinned = Feature(PinnedUnpinnedSharedRootFG.NAME, options={"variant": "a"}, compute_framework="PyArrowTable")

    _, steps = _shared_root_steps([pinned, PinnedUnpinnedSharedRootFG.NAME])

    assert len(steps) == 2
