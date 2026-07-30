import gc
import os
from typing import Any
import pytest

from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.prepare import accessible_plugins
from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer


def _clear_warned_unregistered() -> None:
    """Clear the warn-mode once-per-process dedup set if the implementation provides it."""
    warned = getattr(accessible_plugins, "_warned_unregistered", None)
    if warned is not None:
        warned.clear()


@pytest.fixture(autouse=True)
def restore_default_plugin_registry() -> Any:
    """Snapshot and restore the default plugin registry around every test.

    Also clears the warn-mode dedup set so warn-mode tests stay independent.
    """
    registry = PluginRegistry.default()
    snapshot = registry.snapshot()
    _clear_warned_unregistered()
    yield
    registry.restore(snapshot)
    _clear_warned_unregistered()


@pytest.fixture(autouse=True)
def set_acero_alignment_handling() -> Any:
    """Modern hardware does not care about this. https://arrow.apache.org/docs/cpp/env_vars.html"""
    os.environ["ACERO_ALIGNMENT_HANDLING"] = "ignore"
    yield
    # Optionally, unset the variable or reset it after the test
    del os.environ["ACERO_ALIGNMENT_HANDLING"]


@pytest.fixture(scope="session")
def flight_server_setup() -> ParallelRunnerFlightServer:
    return ParallelRunnerFlightServer()


@pytest.fixture(scope="session")
def flight_server(flight_server_setup: ParallelRunnerFlightServer) -> Any:
    yield flight_server_setup
    flight_server_setup.end_flight_server_process()


NO_GC_FREEZE_ENV_VAR = "MLODA_NO_GC_FREEZE"

# CPython 3.12 seeds the permanent generation with ~375 objects at interpreter startup (3.10, 3.11, 3.13 and 3.14
# seed none), while freezing a realistic imported graph moves tens of thousands. A truthiness check would read
# 3.12's seed as a host freeze and never freeze on that version, so ownership is decided against this floor.
MIN_FROZEN_GRAPH_OBJECTS = 1000

_froze_the_graph = False


# The gc.collect() is load-bearing, not a tidy-up: a FeatureGroup subclass that is dead but not yet collected here
# gets frozen alive and stays visible to get_all_subclasses(FeatureGroup) for the whole run, since objects frozen
# before a collection are never reclaimed afterwards on any of 3.10-3.14. A non-empty permanent generation means a
# host froze its own graph before calling pytest, and that freeze is not ours to touch. Freezing also empties
# gc.get_objects() and hides frozen referrers from gc.get_referrers(), so MLODA_NO_GC_FREEZE opts out of it for
# anyone debugging object retention inside the suite.
def pytest_collection_finish(session: Any) -> None:
    """Freeze once all test modules are imported and no test has run, so later gc.collect() calls skip them."""
    global _froze_the_graph
    if os.getenv(NO_GC_FREEZE_ENV_VAR):
        return
    if gc.get_freeze_count() > MIN_FROZEN_GRAPH_OBJECTS:
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
