"""Failing tests for issue #722 Stage 2 hardening (post-review fix round).

Pins review-mandated hardening of ``mloda.core.resolve.environment``:

1. Error-path payloads are JSON-serializable AND redacted: the raw provider
   message stays on the internal ``message`` field (it feeds the adapter
   re-raise) but never leaks into ``to_payload()``; the payload names the
   category and the exception type instead.
2. Junk members in a ``compute_framework_rule`` set become framework records
   with ``INVALID_DECLARATION`` provenance instead of crashing the build,
   while the legacy ``PreFilterPlugins`` mapping stays tolerant.
3. Carried exceptions are stripped of tracebacks (no frame-graph retention),
   while the adapter still re-raises the original exception type.
4. Framework availability is sampled at most once per framework per build.
5. Snapshots never re-discover, even for a collector-less build
   (TARGET CONTRACT, passes today).
6. ``conflict_identities`` carries no duplicate identities.

RED phase: tests 1, 2, 3a, 3b, 4, and 6 fail against the current
implementation; test 5 passes and pins the target contract.
"""

from __future__ import annotations

import gc
import json
import linecache
import sys
import textwrap
from typing import Any, Iterator, cast

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import PreFilterPlugins, RedefinitionConflictError
from mloda.core.resolve.environment import (
    EnvironmentBuildOutcome,
    EnvironmentProvenance,
    build_resolution_environment,
)
from mloda.core.resolve.identity import PluginIdentity


REDACTION_FEATURE = "probe722e_redacted"
JUNK_FEATURE = "probe722e_junk"
COUNTING_FEATURE = "probe722e_counting"
LATE_FEATURE = "probe722e_late"
REDEF_FEATURE = "probe722e_redef"

# Fake credential-shaped provider message used to prove payload redaction; never a real secret.
_FAKE_SECRET_MESSAGE_722E = "token=super-secret-722e"  # nosec B105


# ---------------------------------------------------------------------------
# Mock compute frameworks (unique names, suffix 722E)
# ---------------------------------------------------------------------------


class CfwValid722E(ComputeFramework):
    """Available framework enabled by most builds in this module."""


# Counts every is_available() call on the counting framework; cleared per test (xdist-safe:
# workers are separate processes, and within one worker the owning test resets it first).
_AVAILABILITY_CALLS_722E: list[str] = []


class CfwCounting722E(ComputeFramework):
    """Available framework whose availability probe counts its own invocations."""

    @staticmethod
    def is_available() -> bool:
        _AVAILABILITY_CALLS_722E.append("call")
        return True


# A declaration member that is NOT a ComputeFramework subclass. It passes the base
# compute_framework_definition plumbing (which only checks the rule is a set) and
# reaches the per-FG classification in build_resolution_environment.
_JUNK_DECLARATION_MEMBER_722E = cast(type[ComputeFramework], 42)


# ---------------------------------------------------------------------------
# Probe feature groups (match ONLY their own probe722e_* names)
# ---------------------------------------------------------------------------


class ProbeCounting722E(FeatureGroup):
    """Probe declaring only the counting framework, for the single-sampling pin."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwCounting722E}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == COUNTING_FEATURE


def _make_secret_raising_fg() -> type[FeatureGroup]:
    """Probe whose compute_framework_rule raises with a credential-shaped message.

    Scoped per test (locals die at return, gc fixture collects after), so the
    broken declaration never leaks into collector-less builds of other tests.
    """
    gc.collect()

    class ProbeSecretDeclaration722E(FeatureGroup):
        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            raise ValueError(_FAKE_SECRET_MESSAGE_722E)

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            return str(feature_name) == REDACTION_FEATURE

    return ProbeSecretDeclaration722E


def _make_junk_declaration_fg() -> type[FeatureGroup]:
    """Probe declaring one valid framework plus a member that is no framework at all.

    Scoped per test like the secret-raising probe, so the junk declaration never
    leaks into collector-less builds of other tests.
    """
    gc.collect()

    class ProbeJunkDeclaration722E(FeatureGroup):
        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {CfwValid722E, _JUNK_DECLARATION_MEMBER_722E}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            return str(feature_name) == JUNK_FEATURE

    return ProbeJunkDeclaration722E


# ---------------------------------------------------------------------------
# Jupyter-cell simulation helpers for the redefinition conflict
# (transplanted from tests/test_core/test_resolution_parity/test_parity_environment.py)
# ---------------------------------------------------------------------------

# Keeps exec'd classes alive across assertions, like IPython's Out[N] history. Leaking them for the process
# lifetime is safe: the autouse cleanup unbinds them from __main__, so they are no longer live-in-module and
# dedup preserves instead of re-raising, and they only match the unique name probe722e_redef.
_REF_STORE: list[Any] = []


def _exec_fg_in_main(class_name: str, body: str, cell_label: str) -> type[FeatureGroup]:
    """Exec a FeatureGroup subclass into __main__ with linecache-backed source (Jupyter cell simulation)."""
    main_mod = sys.modules["__main__"]
    src = textwrap.dedent(body)
    filename = f"<{cell_label}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(keepends=True), filename)
    exec(compile(src, filename, "exec"), main_mod.__dict__)  # nosec B102
    return cast(type[FeatureGroup], main_mod.__dict__[class_name])


def _make_fg_source(class_name: str, feature_name: str, extra_body: str = "") -> str:
    return f"""
from mloda.core.abstract_plugins.feature_group import FeatureGroup as _FG_BASE_

class {class_name}(_FG_BASE_):
    @classmethod
    def feature_names_supported(cls):
        return {{"{feature_name}"}}
{extra_body}
"""


def _make_conflicting_pair(qualname: str, feature_name: str) -> tuple[type[FeatureGroup], type[FeatureGroup]]:
    """Two classes sharing (module, qualname) with DIFFERENT source; v2 is live in __main__."""
    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(qualname, feature_name, extra_body="    def extra_method(self):\n        return 722\n")
    v1 = _exec_fg_in_main(qualname, src_v1, f"cell-{qualname}-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, f"cell-{qualname}-v2")
    _REF_STORE.extend([v1, v2])
    return v1, v2


@pytest.fixture(autouse=True)
def _cleanup_main_module_attrs() -> Iterator[None]:
    """Snapshot __main__ attrs and restore after each test (xdist safety, mirrors test_feature_group_dedup)."""
    main_mod = sys.modules["__main__"]
    snapshot = set(main_mod.__dict__.keys())
    yield
    for key in set(main_mod.__dict__.keys()) - snapshot:
        main_mod.__dict__.pop(key, None)


@pytest.fixture
def _collect_after() -> Iterator[None]:
    """Collect the per-test broken class once the test's frames are gone."""
    yield
    gc.collect()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _off_collector(*probes: type[FeatureGroup]) -> PluginCollector:
    return PluginCollector.enabled_feature_groups(set(probes)).set_strict_mode("off")


def _build_conflict_outcome(qualname: str) -> tuple[EnvironmentBuildOutcome, PluginCollector]:
    """Build the environment over a fresh same-qualname different-source pair."""
    v1, v2 = _make_conflicting_pair(qualname, REDEF_FEATURE)
    collector = PluginCollector.enabled_feature_groups({v1, v2}).set_strict_mode("off")
    outcome = build_resolution_environment(compute_frameworks={CfwValid722E}, plugin_collector=collector)
    return outcome, collector


# ---------------------------------------------------------------------------
# 1. Error-path payload is serializable and redacted
# ---------------------------------------------------------------------------


def test_error_payload_is_serializable_and_redacted(_collect_after: None) -> None:
    """The internal message stays exact for the adapter; the payload never carries the raw provider text."""
    broken = _make_secret_raising_fg()
    collector = _off_collector(broken)

    outcome = build_resolution_environment(compute_frameworks={CfwValid722E}, plugin_collector=collector)

    assert outcome.snapshot is None
    assert len(outcome.errors) == 1
    error = outcome.errors[0]
    assert error.category == "invalid_declaration"
    assert error.message == _FAKE_SECRET_MESSAGE_722E

    dumped = json.dumps(outcome.to_payload())
    assert "super-secret-722e" not in dumped
    assert "object at 0x" not in dumped
    assert "Traceback" not in dumped
    assert "invalid_declaration" in dumped
    assert "ValueError" in dumped


# ---------------------------------------------------------------------------
# 2. Junk declaration members do not crash the build
# ---------------------------------------------------------------------------


def test_junk_declaration_member_is_recorded_not_crashing(_collect_after: None) -> None:
    """A non-framework rule member becomes an INVALID_DECLARATION record; the valid one stays accessible."""
    probe = _make_junk_declaration_fg()
    collector = _off_collector(probe)

    outcome = build_resolution_environment(compute_frameworks={CfwValid722E}, plugin_collector=collector)

    assert outcome.errors == ()
    assert outcome.snapshot is not None
    snapshot = outcome.snapshot

    identity = PluginIdentity.from_class(probe)
    matches = [record for record in snapshot.records if record.identity == identity]
    assert len(matches) == 1
    record = matches[0]

    assert len(record.frameworks) == 2
    invalid = [fw for fw in record.frameworks if fw.provenance is EnvironmentProvenance.INVALID_DECLARATION]
    assert len(invalid) == 1
    accessible = [fw for fw in record.frameworks if fw.provenance is EnvironmentProvenance.ACCESSIBLE]
    assert len(accessible) == 1
    assert accessible[0].identity == PluginIdentity.from_class(CfwValid722E)

    entries = [entry for entry in snapshot.accessible if entry[0] is probe]
    assert len(entries) == 1
    assert entries[0][1] == (CfwValid722E,)

    # Legacy parity (passes today, pinned here to keep the invariant): the adapter
    # tolerates the junk member and maps the probe to the valid framework only.
    mapping = PreFilterPlugins(compute_frameworks={CfwValid722E}, plugin_collector=collector).get_accessible_plugins()
    assert mapping == {probe: {CfwValid722E}}


# ---------------------------------------------------------------------------
# 3. Stored exceptions carry no traceback
# ---------------------------------------------------------------------------


def test_invalid_declaration_exception_carries_no_traceback(_collect_after: None) -> None:
    """The carried provider exception keeps its type but is stripped of its traceback."""
    broken = _make_secret_raising_fg()
    collector = _off_collector(broken)

    outcome = build_resolution_environment(compute_frameworks={CfwValid722E}, plugin_collector=collector)

    assert len(outcome.errors) == 1
    error = outcome.errors[0]
    assert error.category == "invalid_declaration"
    assert error.exception is not None
    assert type(error.exception) is ValueError
    assert error.exception.__traceback__ is None


def test_redefinition_conflict_exception_carries_no_traceback_and_adapter_reraises() -> None:
    """The carried dedup exception is traceback-free while the adapter still raises the original type."""
    outcome, collector = _build_conflict_outcome("ProbeRedefNoTb722E")

    assert outcome.snapshot is None
    conflicts = [error for error in outcome.errors if error.category == "redefinition_conflict"]
    assert len(conflicts) == 1

    with pytest.raises(RedefinitionConflictError):
        PreFilterPlugins(compute_frameworks={CfwValid722E}, plugin_collector=collector)

    assert conflicts[0].exception is not None
    assert isinstance(conflicts[0].exception, RedefinitionConflictError)
    assert conflicts[0].exception.__traceback__ is None


# ---------------------------------------------------------------------------
# 4. is_available sampled at most once per framework per build
# ---------------------------------------------------------------------------


def test_is_available_sampled_at_most_once_per_framework_per_build() -> None:
    """One build consults a framework's availability once; classification reuses the sampled result."""
    collector = _off_collector(ProbeCounting722E)
    _AVAILABILITY_CALLS_722E.clear()

    outcome = build_resolution_environment(compute_frameworks={CfwCounting722E}, plugin_collector=collector)

    assert outcome.snapshot is not None  # premise guard: the build itself succeeds
    assert len(_AVAILABILITY_CALLS_722E) == 1


# ---------------------------------------------------------------------------
# 5. Stronger no-rediscovery pin (TARGET CONTRACT, passes today)
# ---------------------------------------------------------------------------


def test_collectorless_snapshot_ignores_later_class_definitions() -> None:
    """TARGET CONTRACT: even without a collector, a snapshot never re-discovers late classes."""
    gc.collect()  # flush any per-test broken probes lingering from earlier tests in this worker

    outcome = build_resolution_environment(compute_frameworks={CfwValid722E}, plugin_collector=None)
    assert outcome.errors == ()
    assert outcome.snapshot is not None
    snapshot = outcome.snapshot

    records_before = len(snapshot.records)
    mapping_before = snapshot.accessible_mapping()

    def _late_matches(
        cls: type[FeatureGroup],
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == LATE_FEATURE

    late_class = cast(
        type[FeatureGroup],
        type("ProbeLateArrival722E", (FeatureGroup,), {"match_feature_group_criteria": classmethod(_late_matches)}),
    )

    assert len(snapshot.records) == records_before
    assert PluginIdentity.from_class(late_class) not in {record.identity for record in snapshot.records}
    fresh_mapping = snapshot.accessible_mapping()
    assert fresh_mapping == mapping_before
    assert late_class not in fresh_mapping

    del late_class
    gc.collect()


# ---------------------------------------------------------------------------
# 6. Duplicate conflict identities are deduplicated
# ---------------------------------------------------------------------------


def test_conflict_identities_carry_no_duplicates() -> None:
    """conflict_identities lists each conflicting identity once, even for same-identity class pairs."""
    outcome, _collector = _build_conflict_outcome("ProbeRedefDedup722E")

    conflicts = [error for error in outcome.errors if error.category == "redefinition_conflict"]
    assert len(conflicts) == 1
    identities = conflicts[0].conflict_identities
    assert identities != ()
    assert len(identities) == len(set(identities))
