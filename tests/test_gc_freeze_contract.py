"""Pin the gc.freeze contract that keeps the suite's per-test gc.collect() calls off the hot path.

The suite calls gc.collect() constantly (from the shared autouse isolation fixture in tests/conftest.py, and
inline in test bodies and finally blocks), because a throwaway FeatureGroup subclass defined in a test body
lingers in FeatureGroup.__subclasses__() until a GC pass reclaims its cycle. Each collect rescans the whole
imported heap (pandas, polars, duckdb, pyarrow, sklearn, every plugin and test module), so one collect costs
hundreds of milliseconds and gc dominates the suite's wall time. Freezing the imported graph into the
permanent generation once collection has finished makes those collects free without weakening the isolation
they buy, but only if the freeze is owned: gc.unfreeze() is process-global, so the suite may only undo a
freeze it made itself.
"""

from __future__ import annotations

import gc
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup

# The floor the conftest predicate uses to tell its own freeze from a host's, imported rather than copied so
# the predicate and these assertions cannot drift apart.
from tests.conftest import MIN_FROZEN_GRAPH_OBJECTS, NO_GC_FREEZE_ENV_VAR


_THROWAWAY_NAME = "GcFreezeThrowawayFeatureGroup"

_REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE_MARKER = "GC_FREEZE_COUNTS"

_FREEZE_MISSING = (
    "Freezing the suite's imported graph moves tens of thousands of objects into the permanent generation, so a "
    f"count at or below the ownership floor of {MIN_FROZEN_GRAPH_OBJECTS} is not that freeze: CPython 3.12 seeds "
    "~375 objects there at interpreter startup, which is why a count above 0 proves nothing. tests/conftest.py "
    "must call gc.freeze() once pytest collection has finished, so the per-test gc.collect() calls stop "
    "rescanning the whole imported heap; without it a single gc.collect() costs hundreds of milliseconds and the "
    f"suite spends most of its wall time in gc. A count of 0 with {NO_GC_FREEZE_ENV_VAR} set is that opt-out."
)

_UNFREEZE_MISSING = (
    "tests/conftest.py must undo its own gc.freeze() with gc.unfreeze() in pytest_sessionfinish, or the "
    "multiprocessing tests abort with 'Fatal Python error: gilstate_tss_set: failed to set current tstate "
    "(TSS)'. Only an assertion can catch this: under -n both xdist workers abort while the run still exits 0."
)

_OWNERSHIP_VIOLATED = (
    "gc.unfreeze() is process-global. A host process that froze its own graph and then runs this suite "
    "in-process via pytest.main() loses its whole permanent generation, and the xdist controller has the same "
    "shape: it never runs pytest_collection_finish but does run pytest_sessionfinish. tests/conftest.py must "
    "unfreeze only the freeze it made itself."
)

# Only import gc and tests.conftest in the child: the probes must stay fast, and the freeze counts only need
# a realistic imported graph, not the whole plugin universe.
_PAIRED_HOOKS_PROBE = """
import gc

from tests import conftest

baseline = gc.get_freeze_count()
conftest.pytest_collection_finish(session=None)
frozen = gc.get_freeze_count()
conftest.pytest_sessionfinish(session=None, exitstatus=0)
print("{marker}", baseline, frozen, gc.get_freeze_count())
"""

_HOST_FREEZE_PROBE = """
import gc

from tests import conftest

gc.freeze()
host = gc.get_freeze_count()
{hooks}
print("{marker}", host, gc.get_freeze_count())
"""

_SESSION_FINISH = "conftest.pytest_sessionfinish(session=None, exitstatus=0)"

_COLLECTION_FINISH = "conftest.pytest_collection_finish(session=None)"

# Two shapes reach pytest_sessionfinish with a freeze this suite does not own: the xdist controller, which
# returns True from pytest_collection and so never runs pytest_collection_finish, and an embedding host that
# froze before calling pytest.main().
_HOST_HOOK_SEQUENCES: list[str] = [_SESSION_FINISH, f"{_COLLECTION_FINISH}\n{_SESSION_FINISH}"]

_HOST_HOOK_IDS = ["sessionfinish_only", "collection_finish_then_sessionfinish"]


# No local isolation fixture: the shared autouse one in tests/conftest.py already covers this module (#845).
def _define_throwaway_feature_group() -> type[FeatureGroup]:
    """Define a FeatureGroup subclass inside a function body, exactly as the suite's throwaway probes do."""

    class GcFreezeThrowawayFeatureGroup(FeatureGroup):
        pass

    return GcFreezeThrowawayFeatureGroup


def _throwaway_is_registered() -> bool:
    """True while a subclass named _THROWAWAY_NAME is still reachable from FeatureGroup."""
    return any(cls.__name__ == _THROWAWAY_NAME for cls in get_all_subclasses(FeatureGroup))


def _freeze_counts(probe: str) -> list[int]:
    """Run probe in a fresh interpreter rooted at the repo and return the freeze counts it printed.

    A subprocess, not this process: driving the conftest hooks in-process would unfreeze the live session and
    make the sibling freeze tests order-dependent. The child never inherits the freeze opt-out, so these probes
    measure the hooks themselves even when the ambient run is debugging object retention.
    """
    child_env = {name: value for name, value in os.environ.items() if name != NO_GC_FREEZE_ENV_VAR}
    # Safe: fixed argv (sys.executable plus a probe built from module-level constants), no shell, no user input.
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", probe],
        cwd=str(_REPO_ROOT),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"The freeze probe died in a fresh interpreter (returncode {result.returncode}).\n"
        f"probe:\n{probe}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    marked = [line for line in result.stdout.splitlines() if line.startswith(_PROBE_MARKER)]
    assert len(marked) == 1, f"Expected one {_PROBE_MARKER} line, got {marked}.\nstdout:\n{result.stdout}"
    return [int(value) for value in marked[0].split()[1:]]


def test_import_graph_is_frozen_while_tests_run() -> None:
    """The imported graph sits in the permanent generation for the whole run, so gc.collect() never rescans it."""
    frozen = gc.get_freeze_count()

    assert frozen > MIN_FROZEN_GRAPH_OBJECTS, f"gc.get_freeze_count() is {frozen}. {_FREEZE_MISSING}"


def test_gc_collect_still_reaps_throwaway_feature_groups_under_a_frozen_graph() -> None:
    """Freezing must not extend the permanent generation to classes created later: one gc.collect() still reaps."""
    frozen = gc.get_freeze_count()
    assert frozen > MIN_FROZEN_GRAPH_OBJECTS, f"gc.get_freeze_count() is {frozen}. {_FREEZE_MISSING}"

    # Only booleans stay in this frame across the collect: a retained class reference, or an assert-rewrite
    # temporary holding one, would pin the very cycle whose reclamation is under test. The positive control
    # runs the same probe as the measurement, so a drift between _THROWAWAY_NAME and the class name fails the
    # control instead of silently making the measurement vacuous.
    control = _define_throwaway_feature_group()
    registered_while_referenced = _throwaway_is_registered()
    del control
    gc.collect()
    still_registered = _throwaway_is_registered()

    assert registered_while_referenced, (
        f"Positive control failed: no subclass named {_THROWAWAY_NAME} is reachable from "
        "get_all_subclasses(FeatureGroup) even while one is strongly referenced, so this test can no longer "
        "detect a leak. Fix the probe, not the freeze hook."
    )
    assert not still_registered, (
        f"{_THROWAWAY_NAME} survived gc.collect() and stays in get_all_subclasses(FeatureGroup). Every autouse "
        "isolation fixture in the suite depends on gc.collect() reaping throwaway subclasses; freezing the "
        "import graph must leave classes created after the freeze collectable."
    )


def test_conftest_unfreezes_the_graph_at_session_finish() -> None:
    """The hook pair is balanced: what pytest_collection_finish freezes, pytest_sessionfinish hands back."""
    baseline, frozen, released = _freeze_counts(_PAIRED_HOOKS_PROBE.format(marker=_PROBE_MARKER))

    assert frozen - baseline > MIN_FROZEN_GRAPH_OBJECTS, (
        f"pytest_collection_finish moved {frozen - baseline} objects into the permanent generation "
        f"(count {baseline} before, {frozen} after). {_FREEZE_MISSING}"
    )
    # Not `released == 0`: on CPython 3.12 the ~375 objects the interpreter seeds at startup come back into the
    # permanent generation on the next full collection. Requiring the bulk back holds on 3.10 through 3.14.
    assert released * 10 < frozen, (
        f"pytest_sessionfinish left {released} of {frozen} objects frozen (count {baseline} before the freeze), "
        f"so it did not release the graph it froze. {_UNFREEZE_MISSING}"
    )


@pytest.mark.parametrize("hooks", _HOST_HOOK_SEQUENCES, ids=_HOST_HOOK_IDS)
def test_conftest_never_unfreezes_a_freeze_it_does_not_own(hooks: str) -> None:
    """A freeze made before the suite ran survives pytest_sessionfinish, whether or not collection finished."""
    host, remaining = _freeze_counts(_HOST_FREEZE_PROBE.format(hooks=hooks, marker=_PROBE_MARKER))

    assert host > MIN_FROZEN_GRAPH_OBJECTS, (
        f"Positive control failed: the host's own gc.freeze() left only {host} objects in the permanent "
        f"generation, at or below the {MIN_FROZEN_GRAPH_OBJECTS} floor the conftest reads as an empty one, so "
        "this probe no longer exercises the ownership path at all."
    )
    assert remaining >= host // 2, (
        f"The host froze {host} objects and {remaining} are left after the conftest hooks ran. {_OWNERSHIP_VIOLATED}"
    )
