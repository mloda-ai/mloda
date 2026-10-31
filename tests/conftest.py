import gc
import importlib.util
import os
from collections.abc import Callable, Iterator
from typing import Any
import pytest

from mloda.core.abstract_plugins.components import utils
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.provider import MatchRejection, collect_match_rejections
from mloda.core.abstract_plugins.implementation import Implementation
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.prepare import accessible_plugins
from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)

from tests.registry_isolation import (
    IMPLEMENTATIONS,
    reclaim_released_implementations,
    reclaim_since,
    snapshot_roots,
    young_window,
)

_ALL_MP_PRELOAD_MODULES = [
    "mloda.user",
    "mloda.provider",
    "mloda.core.runtime.worker.multiprocessing_worker",
    "mloda.core.core.cfw_manager",
    "mloda.core.runtime.flight.flight_server",
    "mloda_plugins.compute_framework.base_implementations.pandas.dataframe",
    "mloda_plugins.compute_framework.base_implementations.pyarrow.table",
    "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework",
    "pandas",
    "pyarrow",
    "pyarrow.flight",
    "pyiceberg.table",
    "pyspark.sql",
]
_OPTIONAL_BACKEND_PACKAGES = {"pyarrow", "pandas", "pyiceberg", "pyspark"}
MP_PRELOAD_MODULES = [
    m
    for m in _ALL_MP_PRELOAD_MODULES
    if m.split(".")[0] not in _OPTIONAL_BACKEND_PACKAGES or importlib.util.find_spec(m.split(".")[0]) is not None
]
# Must run before anything starts a worker process.
os.environ.setdefault("MLODA_MP_START_METHOD", "forkserver")
os.environ.setdefault("MLODA_MP_PRELOAD", ",".join(MP_PRELOAD_MODULES))
# OPENBLAS_NUM_THREADS=1 keeps numpy's thread pool out of the forkserver; pyarrow's jemalloc thread is fork-safe
# (jemalloc's atfork handlers reset its state in the child); polars and duckdb stay out of the preload.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# Autouse fixtures in one module are ordered by name, and this one sorts first, so it tears down last.
@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution(request: pytest.FixtureRequest) -> Any:
    """Fail a test leaving its own FeatureGroup, BaseTransformer or Implementation subclass registered (#845).

    ComputeFramework and Extender subclasses are reclaimed too but never reported.
    """
    snapshot = snapshot_roots()
    module_name = request.module.__name__
    with young_window():
        yield
        leaked = reclaim_since(snapshot, module_name)
    assert not leaked, (
        f"Leaked FeatureGroup/BaseTransformer/Implementation subclasses from {module_name}: {[c.__name__ for c in leaked]}"
    )


def _clear_warned_unregistered() -> None:
    """Clear the warn-mode once-per-process dedup set if the implementation provides it."""
    warned = getattr(accessible_plugins, "_warned_unregistered", None)
    if warned is not None:
        warned.clear()


def _clear_warn_once_for_registries() -> None:
    """Clear safe_field's warn_once_for dedup registries if the implementation provides them."""
    weak_registry = getattr(utils, "_warn_once_for_weak", None)
    if weak_registry is not None:
        weak_registry.clear()
    strong_registry = getattr(utils, "_warn_once_for_strong", None)
    if strong_registry is not None:
        strong_registry.clear()


@pytest.fixture(autouse=True)
def restore_default_plugin_registry() -> Any:
    """Snapshot and restore the default plugin registry around every test.

    Also clears the warn-mode dedup set and safe_field's warn_once_for registries so
    warn-mode tests and once-per-process WARNING dedup tests stay independent.
    """
    registry = PluginRegistry.default()
    snapshot = registry.snapshot()
    _clear_warned_unregistered()
    _clear_warn_once_for_registries()
    yield
    registry.restore(snapshot)
    _clear_warned_unregistered()
    _clear_warn_once_for_registries()


@pytest.fixture(autouse=True)
def restore_dynamic_feature_group_cache(_no_feature_group_registry_pollution: None) -> Any:
    """Restore DynamicFeatureGroupCreator's class cache, so classes a test cached do not read as leaks."""
    before = dict(DynamicFeatureGroupCreator._created_classes)
    yield
    DynamicFeatureGroupCreator._created_classes.clear()
    DynamicFeatureGroupCreator._created_classes.update(before)


@pytest.fixture(autouse=True)
def set_acero_alignment_handling() -> Any:
    """Modern hardware does not care about this. https://arrow.apache.org/docs/cpp/env_vars.html"""
    os.environ["ACERO_ALIGNMENT_HANDLING"] = "ignore"
    yield
    # Optionally, unset the variable or reset it after the test
    del os.environ["ACERO_ALIGNMENT_HANDLING"]


@pytest.fixture
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a MATCH_REJECTION_REASONS window around one matcher call, mirroring the engine's per-candidate window."""
    with collect_match_rejections() as window:
        yield window


@pytest.fixture(scope="session")
def flight_server_setup() -> ParallelRunnerFlightServer:
    return ParallelRunnerFlightServer()


@pytest.fixture(scope="session")
def flight_server(flight_server_setup: ParallelRunnerFlightServer) -> Any:
    yield flight_server_setup
    flight_server_setup.end_flight_server_process()


NO_GC_FREEZE_ENV_VAR = "MLODA_NO_GC_FREEZE"

# CPython 3.12 seeds the permanent generation with ~375 objects at interpreter startup (3.11, 3.13 and 3.14
# seed none), while freezing a realistic imported graph moves tens of thousands. A truthiness check would read
# 3.12's seed as a host freeze and never freeze on that version, so ownership is decided against this floor.
MIN_FROZEN_GRAPH_OBJECTS = 1000

_froze_the_graph = False


def _suite_will_freeze() -> bool:
    return not os.getenv(NO_GC_FREEZE_ENV_VAR) and gc.get_freeze_count() <= MIN_FROZEN_GRAPH_OBJECTS


# Imports every plugin module, and the backends they import at module level, before the freeze; only the default
# registry is restored, so tests start from the same registry state. A separate hook keeps the contract test's fast
# probes from paying the import.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session: Any, config: Any, items: Any) -> None:
    # Runs before -m deselection: MULTIPROCESSING-parametrized cases spawn workers and belong to `tox -e subprocess`.
    for item in items:
        for value in getattr(getattr(item, "callspec", None), "params", {}).values():
            members = value if isinstance(value, (set, frozenset, list, tuple)) else (value,)
            if any(m is ParallelizationMode.MULTIPROCESSING for m in members):
                item.add_marker(pytest.mark.subprocess)
                break
    if not _suite_will_freeze():
        return
    snapshot = PluginRegistry.default().snapshot()
    PluginLoader.all()
    PluginRegistry.default().restore(snapshot)


# The gc.collect() is load-bearing, not a tidy-up: a FeatureGroup subclass that is dead but not yet collected here
# gets frozen alive and stays visible to get_all_subclasses(FeatureGroup) for the whole run, since objects frozen
# before a collection are never reclaimed afterwards on any of 3.11-3.14. A non-empty permanent generation means a
# host froze its own graph before calling pytest, and that freeze is not ours to touch. Freezing also empties
# gc.get_objects() and hides frozen referrers from gc.get_referrers(), so MLODA_NO_GC_FREEZE opts out of it for
# anyone debugging object retention inside the suite.
def pytest_collection_finish(session: Any) -> None:
    """Freeze once all test modules are imported and no test has run, so later gc.collect() calls skip them."""
    global _froze_the_graph
    if not _suite_will_freeze():
        return
    gc.collect()
    gc.freeze()
    _froze_the_graph = True


# tryfirst because hook impls run in reverse registration order: a deeper conftest or a later-registered plugin
# that raises in pytest_sessionfinish would skip this unfreeze. A missed unfreeze surfaces as 'Fatal Python error:
# gilstate_tss_set: failed to set current tstate (TSS)', which exits 134 serially but stays invisible under xdist,
# where both workers abort and the run still exits 0.
@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    """Unfreeze our own freeze only, or the multiprocessing tests abort with a gilstate_tss_set fatal error."""
    global _froze_the_graph
    if not _froze_the_graph:
        return
    gc.unfreeze()
    _froze_the_graph = False


_terminal_reporter: Any = None


def pytest_sessionstart(session: Any) -> None:
    global _terminal_reporter
    if hasattr(session.config, "workerinput"):
        _terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")


def drop_passing_report(stats: dict[str, list[Any]], report: Any) -> None:
    """Pop a passing report that was just appended as the last entry of its stats lists."""
    if not report.passed:
        return
    for key in ("", "passed"):
        reports = stats.get(key)
        if reports and reports[-1] is report:
            reports.pop()


# A worker's terminal reporter keeps every report for a summary only the controller prints, and those reports grow
# the unfrozen heap that each full gc.collect() traverses.
@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report: Any) -> None:
    if _terminal_reporter is None:
        return
    drop_passing_report(_terminal_reporter.stats, report)


CHECK_SKIP_COUNT_ENV_VAR = "CHECK_SKIP_COUNT"
EXPECTED_SKIP_COUNT_ENV_VAR = "EXPECTED_SKIP_COUNT"

if os.getenv(CHECK_SKIP_COUNT_ENV_VAR) == "1":

    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(terminalreporter: Any, exitstatus: Any, config: Any) -> None:
        expected_skips = os.getenv(EXPECTED_SKIP_COUNT_ENV_VAR)
        if expected_skips is None:
            raise SystemExit(f"ERROR: {EXPECTED_SKIP_COUNT_ENV_VAR} is not set.")

        try:
            int_expected_skips = int(expected_skips)
        except ValueError:
            raise SystemExit(f"ERROR: {EXPECTED_SKIP_COUNT_ENV_VAR} must be an integer.")

        skipped = len(terminalreporter.stats.get("skipped", []))
        if skipped != int_expected_skips:
            raise SystemExit(
                f"""ERROR: Expected {expected_skips} skipped tests, but got {skipped}. Somehow the number of skipped tests does not match the expected value. Please check your test setup.
                    If this expected adjust the var EXPECTED_SKIP_COUNT in the tox.ini. 
                    If this just during development, you can adjust CHECK_SKIP_COUNT to something else than 1.
                    """
            )


@pytest.fixture
def make_implementation() -> Iterator[Callable[..., type[Implementation[Any]]]]:
    """make(group, framework, fn=None, base=Implementation, name=None); teardown asserts every class is collected."""
    mark = IMPLEMENTATIONS.mark()

    def make(
        group: Any,
        framework: Any,
        fn: Callable[[Any, Any, Any], Any] | None = None,
        base: type[Implementation[Any]] = Implementation,
        name: str | None = None,
    ) -> type[Implementation[Any]]:
        class _Impl(base):  # type: ignore[misc, valid-type]
            feature_group = group
            compute_framework = framework

            @classmethod
            def calculate(cls, serving: Any, data: Any, features: Any, framework: Any) -> Any:
                return None if fn is None else fn(serving, data, features)

        if name is not None:
            _Impl.__name__ = name
            _Impl.__qualname__ = name
        return _Impl

    yield make
    leaked = reclaim_released_implementations(mark, __name__)
    assert not leaked, f"leaked implementations: {leaked}"
