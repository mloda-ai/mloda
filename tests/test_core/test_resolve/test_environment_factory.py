"""Failing tests for issue #722 Stage 2: the immutable environment state API.

Defines the contract of ``mloda.core.resolve.environment``:
``build_resolution_environment`` runs the same classification pipeline as
``PreFilterPlugins`` (collector applicability, strict mode via injected registry,
dedup, availability, run-framework intersection) but returns structured state
instead of raising. On the conditions where ``PreFilterPlugins`` raises, the
outcome carries errors with the exact current message text and no snapshot.
``PreFilterPlugins`` becomes an adapter over this factory with unchanged
raising behavior.

RED phase: ``mloda.core.resolve`` does not exist yet, so this module fails at
collection with ``ModuleNotFoundError``.
"""

from __future__ import annotations

import dataclasses
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
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.prepare.accessible_plugins import PreFilterPlugins, RedefinitionConflictError
from mloda.core.resolve.environment import (
    EnvironmentBuildOutcome,
    EnvironmentProvenance,
    FeatureGroupRecord,
    FrameworkRecord,
    ResolutionEnvironmentSnapshot,
    build_resolution_environment,
)
from mloda.core.resolve.identity import PluginIdentity


ACCESSIBLE_FEATURE = "probe722d_accessible"
DISABLED_FEATURE = "probe722d_disabled"
REGISTERED_A_FEATURE = "probe722d_registered_a"
UNREGISTERED_B_FEATURE = "probe722d_unregistered_b"
THREE_FW_FEATURE = "probe722d_threefw"
STRICT_ONLY_FEATURE = "probe722d_strict_only"
TWO_FW_FEATURE = "probe722d_twofw"
BROKEN_FEATURE = "probe722d_broken"
REDEF_FEATURE = "probe722d_redef"


# ---------------------------------------------------------------------------
# Mock compute frameworks (unique names, suffix 722D)
# ---------------------------------------------------------------------------


class CfwAvail722D(ComputeFramework):
    """Available framework enabled for most builds in this module."""


class CfwSecond722D(ComputeFramework):
    """Available framework enabled only by the fingerprint-sensitivity build."""


class CfwNotEnabled722D(ComputeFramework):
    """Available framework never passed into a build in this module."""


class CfwUnavailable722D(ComputeFramework):
    """Framework whose backing dependency is never installed."""

    @staticmethod
    def is_available() -> bool:
        return False


class CfwStrictRun722D(ComputeFramework):
    """Available framework carrying the strict-mode builds."""


# ---------------------------------------------------------------------------
# Probe feature groups (match ONLY their own probe722d_* names)
# ---------------------------------------------------------------------------


class ProbeAccessible722D(FeatureGroup):
    """Collector-enabled probe with one available, run-enabled framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAvail722D}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == ACCESSIBLE_FEATURE


class ProbeDisabled722D(FeatureGroup):
    """Probe deliberately left out of the collector's enabled set."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAvail722D}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == DISABLED_FEATURE


class ProbeRegisteredA722D(FeatureGroup):
    """Probe registered in the injected strict-mode registry."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwStrictRun722D}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == REGISTERED_A_FEATURE


class ProbeUnregisteredB722D(FeatureGroup):
    """Probe NOT registered in the injected strict-mode registry."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwStrictRun722D}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == UNREGISTERED_B_FEATURE


class ProbeThreeFw722D(FeatureGroup):
    """Declares one enabled, one unavailable, and one not-enabled framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAvail722D, CfwUnavailable722D, CfwNotEnabled722D}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == THREE_FW_FEATURE


class ProbeStrictOnly722D(FeatureGroup):
    """Concrete probe that is never registered; strict mode drops it fatally."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwStrictRun722D}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == STRICT_ONLY_FEATURE


class ProbeTwoFw722D(FeatureGroup):
    """Declares two available frameworks for the fingerprint-sensitivity test."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAvail722D, CfwSecond722D}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == TWO_FW_FEATURE


def _make_broken_declaration_fg() -> type[FeatureGroup]:
    """Probe whose compute_framework_definition raises when consulted.

    Scoped per test (locals die at return, gc fixture collects after), so the
    broken declaration never leaks into collector-less builds of other tests.
    """
    gc.collect()

    class ProbeBrokenDeclaration722D(FeatureGroup):
        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            raise ValueError("broken definition 722d")

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            return str(feature_name) == BROKEN_FEATURE

    return ProbeBrokenDeclaration722D


# ---------------------------------------------------------------------------
# Jupyter-cell simulation helpers for the redefinition conflict
# (transplanted from tests/test_core/test_resolution_parity/test_parity_environment.py)
# ---------------------------------------------------------------------------

# Keeps exec'd classes alive across assertions, like IPython's Out[N] history. Leaking them for the process
# lifetime is safe: the autouse cleanup unbinds them from __main__, so they are no longer live-in-module and
# dedup preserves instead of re-raising, and they only match the unique name probe722d_redef.
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


def _strict_collector(registry: PluginRegistry, *probes: type[FeatureGroup]) -> PluginCollector:
    return PluginCollector.enabled_feature_groups(set(probes)).set_strict_mode("strict").set_registry(registry)


def _snapshot_or_fail(outcome: EnvironmentBuildOutcome) -> ResolutionEnvironmentSnapshot:
    assert outcome.errors == ()
    assert outcome.snapshot is not None
    return outcome.snapshot


def _record_for(snapshot: ResolutionEnvironmentSnapshot, plugin_class: type[FeatureGroup]) -> FeatureGroupRecord:
    identity = PluginIdentity.from_class(plugin_class)
    matches = [record for record in snapshot.records if record.identity == identity]
    assert len(matches) == 1, f"expected exactly one record for {identity}, got {len(matches)}"
    return matches[0]


def _framework_record(record: FeatureGroupRecord, framework: type[ComputeFramework]) -> FrameworkRecord:
    identity = PluginIdentity.from_class(framework)
    matches = [framework_record for framework_record in record.frameworks if framework_record.identity == identity]
    assert len(matches) == 1, f"expected exactly one framework record for {identity}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Provenance classification
# ---------------------------------------------------------------------------


def test_accessible_probe_is_classified_accessible() -> None:
    """A collector-enabled probe with an available run-enabled framework is ACCESSIBLE end to end."""
    collector = _off_collector(ProbeAccessible722D)

    outcome = build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)
    snapshot = _snapshot_or_fail(outcome)

    record = _record_for(snapshot, ProbeAccessible722D)
    assert record.provenance is EnvironmentProvenance.ACCESSIBLE
    assert EnvironmentProvenance.ACCESSIBLE.value == "accessible"
    assert record.identity.render() == f"{ProbeAccessible722D.__module__}:{ProbeAccessible722D.__qualname__}"

    assert _framework_record(record, CfwAvail722D).provenance is EnvironmentProvenance.ACCESSIBLE

    accessible_entries = [entry for entry in snapshot.accessible if entry[0] is ProbeAccessible722D]
    assert len(accessible_entries) == 1
    assert accessible_entries[0][1] == (CfwAvail722D,)

    assert snapshot.strict_mode == "off"
    assert snapshot.requested_frameworks == (PluginIdentity.from_class(CfwAvail722D),)
    assert snapshot.enabled_frameworks == (PluginIdentity.from_class(CfwAvail722D),)


def test_collector_excluded_probe_is_disabled_by_collector() -> None:
    """A probe excluded by the collector is DISABLED_BY_COLLECTOR and absent from accessible."""
    collector = _off_collector(ProbeAccessible722D)

    snapshot = _snapshot_or_fail(
        build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)
    )

    record = _record_for(snapshot, ProbeDisabled722D)
    assert record.provenance is EnvironmentProvenance.DISABLED_BY_COLLECTOR
    assert all(entry[0] is not ProbeDisabled722D for entry in snapshot.accessible)
    assert ProbeDisabled722D not in snapshot.accessible_mapping()


def test_strict_mode_rejects_unregistered_probe_as_policy_rejected() -> None:
    """Strict mode with an injected registry rejects only the unregistered probe, without a fatal error."""
    registry = PluginRegistry()
    registry.register(ProbeRegisteredA722D)
    registry.register(CfwStrictRun722D)
    collector = _strict_collector(registry, ProbeRegisteredA722D, ProbeUnregisteredB722D)

    outcome = build_resolution_environment(compute_frameworks={CfwStrictRun722D}, plugin_collector=collector)
    snapshot = _snapshot_or_fail(outcome)

    assert _record_for(snapshot, ProbeUnregisteredB722D).provenance is EnvironmentProvenance.POLICY_REJECTED
    assert _record_for(snapshot, ProbeRegisteredA722D).provenance is EnvironmentProvenance.ACCESSIBLE
    assert snapshot.strict_mode == "strict"

    mapping = snapshot.accessible_mapping()
    assert mapping.get(ProbeRegisteredA722D) == {CfwStrictRun722D}
    assert ProbeUnregisteredB722D not in mapping


def test_framework_records_distinguish_unavailable_and_not_enabled() -> None:
    """Framework records split per cause: accessible vs unavailable vs not enabled for this run."""
    assert CfwUnavailable722D.is_available() is False  # premise guard
    assert CfwNotEnabled722D.is_available() is True  # premise guard

    collector = _off_collector(ProbeThreeFw722D)
    snapshot = _snapshot_or_fail(
        build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)
    )

    record = _record_for(snapshot, ProbeThreeFw722D)
    assert _framework_record(record, CfwAvail722D).provenance is EnvironmentProvenance.ACCESSIBLE
    assert _framework_record(record, CfwUnavailable722D).provenance is EnvironmentProvenance.UNAVAILABLE
    assert _framework_record(record, CfwNotEnabled722D).provenance is EnvironmentProvenance.NOT_ENABLED

    assert snapshot.accessible_mapping()[ProbeThreeFw722D] == {CfwAvail722D}


# ---------------------------------------------------------------------------
# Fatal parity with PreFilterPlugins (structured errors, exact message text)
# ---------------------------------------------------------------------------


def test_invalid_declaration_is_structured_and_prefilter_raises_same_error(_collect_after: None) -> None:
    """A broken compute_framework_definition becomes a structured invalid_declaration error."""
    broken = _make_broken_declaration_fg()
    collector = PluginCollector.enabled_feature_groups({broken}).set_strict_mode("off")

    with pytest.raises(ValueError, match="broken definition 722d") as excinfo:
        PreFilterPlugins(compute_frameworks={CfwAvail722D}, plugin_collector=collector)

    outcome = build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)

    assert outcome.snapshot is None
    invalid = [error for error in outcome.errors if error.category == "invalid_declaration"]
    assert len(invalid) == 1
    assert invalid[0].message == str(excinfo.value)


def test_strict_mode_fatal_error_matches_prefilter_message() -> None:
    """Strict mode dropping every concrete probe yields the exact PreFilterPlugins message, structured."""
    collector = _strict_collector(PluginRegistry(), ProbeStrictOnly722D)

    with pytest.raises(ValueError, match="Strict mode filtered out all FeatureGroups") as excinfo:
        PreFilterPlugins(compute_frameworks={CfwStrictRun722D}, plugin_collector=collector)

    outcome = build_resolution_environment(compute_frameworks={CfwStrictRun722D}, plugin_collector=collector)

    assert outcome.snapshot is None
    fatal = [error for error in outcome.errors if error.category == "strict_mode_feature_groups"]
    assert len(fatal) == 1
    assert fatal[0].message == str(excinfo.value)


def test_redefinition_conflict_is_structured_and_prefilter_raises() -> None:
    """A same-qualname different-source pair yields a redefinition_conflict error carrying identities."""
    v1, v2 = _make_conflicting_pair("ProbeRedefEnv722D", REDEF_FEATURE)
    collector = PluginCollector.enabled_feature_groups({v1, v2}).set_strict_mode("off")

    outcome = build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)

    assert outcome.snapshot is None
    conflicts = [error for error in outcome.errors if error.category == "redefinition_conflict"]
    assert len(conflicts) == 1
    assert conflicts[0].conflict_identities != ()
    assert all(identity.qualname == "ProbeRedefEnv722D" for identity in conflicts[0].conflict_identities)

    with pytest.raises(RedefinitionConflictError):
        PreFilterPlugins(compute_frameworks={CfwAvail722D}, plugin_collector=collector)


# ---------------------------------------------------------------------------
# Mapping parity, determinism, fingerprint
# ---------------------------------------------------------------------------


def test_accessible_mapping_matches_prefilter_plugins() -> None:
    """On a healthy environment the snapshot mapping equals PreFilterPlugins' mapping exactly."""
    collector = _off_collector(ProbeAccessible722D, ProbeThreeFw722D)

    snapshot = _snapshot_or_fail(
        build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)
    )
    expected = PreFilterPlugins(compute_frameworks={CfwAvail722D}, plugin_collector=collector).get_accessible_plugins()

    assert snapshot.accessible_mapping() == expected


def test_building_twice_is_deterministic() -> None:
    """Identical inputs yield equal record tuples and equal 64-char sha256 hex fingerprints."""
    collector = _off_collector(ProbeAccessible722D, ProbeThreeFw722D)

    first = _snapshot_or_fail(
        build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)
    )
    second = _snapshot_or_fail(
        build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)
    )

    assert first.records == second.records
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert set(first.fingerprint) <= set("0123456789abcdef")


def test_fingerprint_changes_with_second_enabled_framework() -> None:
    """Enabling a second run framework changes the environment fingerprint."""
    collector = _off_collector(ProbeTwoFw722D)

    narrow = _snapshot_or_fail(
        build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)
    )
    wide = _snapshot_or_fail(
        build_resolution_environment(compute_frameworks={CfwAvail722D, CfwSecond722D}, plugin_collector=collector)
    )

    assert narrow.fingerprint != wide.fingerprint


# ---------------------------------------------------------------------------
# Serialization and immutability
# ---------------------------------------------------------------------------


def test_outcome_payload_is_json_serializable_without_class_objects() -> None:
    """to_payload() is plain data: json.dumps succeeds and no class reprs leak through."""
    collector = _off_collector(ProbeAccessible722D, ProbeThreeFw722D)

    outcome = build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)
    dumped = json.dumps(outcome.to_payload())

    assert "<class" not in dumped
    assert "object at 0x" not in dumped


def test_accessible_mapping_returns_fresh_dict_and_snapshot_is_frozen() -> None:
    """Mutating a returned mapping never leaks into the snapshot; fields cannot be reassigned."""
    collector = _off_collector(ProbeAccessible722D)
    snapshot = _snapshot_or_fail(
        build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)
    )

    mapping = snapshot.accessible_mapping()
    assert ProbeAccessible722D in mapping
    mapping.pop(ProbeAccessible722D)

    assert ProbeAccessible722D in snapshot.accessible_mapping()

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(snapshot, "fingerprint", "forged-722d")


def test_snapshot_ignores_later_class_definitions() -> None:
    """A snapshot never re-discovers: a class defined after the build stays invisible to it."""
    collector = _off_collector(ProbeAccessible722D)
    snapshot = _snapshot_or_fail(
        build_resolution_environment(compute_frameworks={CfwAvail722D}, plugin_collector=collector)
    )
    records_before = snapshot.records
    mapping_before = snapshot.accessible_mapping()

    def _late_matches(
        cls: type[FeatureGroup],
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == ACCESSIBLE_FEATURE

    late_class = cast(
        type[FeatureGroup],
        type("ProbeLateArrival722D", (FeatureGroup,), {"match_feature_group_criteria": classmethod(_late_matches)}),
    )

    assert snapshot.records == records_before
    assert PluginIdentity.from_class(late_class) not in {record.identity for record in snapshot.records}
    assert snapshot.accessible_mapping() == mapping_before
    assert late_class not in snapshot.accessible_mapping()

    del late_class
    gc.collect()
