"""Failing target-contract tests for the whole-request diagnostics/report layer (issue #722 Stage 4a).

TARGET API (does not exist yet; every test here FAILS until the Green implementation lands):

mloda.core.resolve.report:
- @dataclass(frozen=True) FeatureResolutionRecord(dependency_path: tuple[str, ...], outcome: ResolutionOutcome)
- @dataclass(frozen=True) ResolutionReport(environment: EnvironmentBuildOutcome | None,
  features: tuple[FeatureResolutionRecord, ...], complete: bool) with to_payload() -> plain JSON data.

mlodaAPI (mloda/core/api/request.py):
- classmethod diagnose(...prepare signature family...) -> ResolutionReport, NON-RAISING for
  resolution/environment failures.
- instance method resolution_report() -> ResolutionReport on a prepared session, taken from the
  engine's planning pass without re-matching.
- FeatureResolutionError raised through prepare()/run_all() engine construction additionally
  carries .report (a ResolutionReport with the partial records, complete=False).

PINNED CHOICES beyond the issue brief (report these to the Green agent):
- Records are ordered in PLANNING ORDER and dependency_path INCLUDES the feature's own name,
  exactly like Engine.resolution_outcomes today: root (name,), child (root, child).
- On a successful diagnose()/resolution_report(), ``environment`` is NOT None: it carries an
  EnvironmentBuildOutcome with a snapshot and no errors (the report describes the whole request).
- ResolutionReport.to_payload() keys: "environment" (EnvironmentBuildOutcome.to_payload() or None),
  "features" (list of {"dependency_path": list[str], "outcome": ResolutionOutcome.to_payload()}),
  "complete" (bool).
- Engine-path environment failures keep today's raise types: plain ValueError for
  strict-mode-filtered-all (pinned via pytest.raises(ValueError, match=...), so a
  FeatureResolutionError subclass upgrade stays allowed) and RedefinitionConflictError for a
  redefinition conflict (pinned exactly; no report assertions on that path).

Probe names use the unique probe722i_ prefix; probe classes carry the 722I suffix so
process-global scans in other tests never trip these fixtures.
"""

from __future__ import annotations

import json
import linecache
import sys
import textwrap
from typing import Any, Iterator, cast

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.api.plan_info import PlanStep
from mloda.core.api.request import mlodaAPI
from mloda.core.prepare.accessible_plugins import RedefinitionConflictError
from mloda.core.resolve.environment import EnvironmentBuildOutcome, ResolutionEnvironmentSnapshot
from mloda.core.resolve.outcome import FeatureResolutionError, ResolutionOutcome, ResolutionStatus
from mloda.core.resolve.report import FeatureResolutionRecord, ResolutionReport
from mloda.core.resolve.request import ResolutionRequestSnapshot
from mloda.core.resolve.resolver import FeatureGroupResolver
from mloda.provider import BaseInputData, DataCreator, FeatureSet
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


BASE_FEATURE = "probe722i_base"
DERIVED_FEATURE = "probe722i_derived"
MISSING_FEATURE = "probe722i_missing"
STRICT_FEATURE = "probe722i_strict"
REDEF_FEATURE = "probe722i_redef"

STRICT_MODE_TEXT = "Strict mode filtered out all FeatureGroups"
NO_FEATURE_GROUPS_TEXT = "No feature groups found"


# ---------------------------------------------------------------------------
# Probe fixtures (DataCreator-backed, PluginCollector-scoped)
# ---------------------------------------------------------------------------


class SourceBase722I(FeatureGroup):
    """Root source for the report probes; matches ONLY probe722i_base via its DataCreator."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={BASE_FEATURE})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {BASE_FEATURE: [1, 2, 3]}


class DerivedOnBase722I(FeatureGroup):
    """Derived probe with an input_features dependency on the base source."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == DERIVED_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(BASE_FEATURE)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {DERIVED_FEATURE: [2, 4, 6]}


class CfwStrict722I(ComputeFramework):
    """Available framework carrying the strict-mode probe."""


class ProbeStrict722I(FeatureGroup):
    """Concrete probe that is never registered in the injected empty registry."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwStrict722I}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == STRICT_FEATURE


def _success_collector() -> PluginCollector:
    return PluginCollector.enabled_feature_groups({SourceBase722I, DerivedOnBase722I})


def _success_features() -> list[Feature | str]:
    return [Feature(BASE_FEATURE), Feature(DERIVED_FEATURE)]


def _strict_collector_with_empty_registry() -> PluginCollector:
    """Enable only the strict probe, strict mode on, EMPTY injected registry (global registry untouched)."""
    collector = PluginCollector.enabled_feature_groups({ProbeStrict722I})
    return collector.set_strict_mode("strict").set_registry(PluginRegistry())


# ---------------------------------------------------------------------------
# Jupyter-cell simulation helpers for the redefinition-conflict path
# (construction pattern mirrors tests/test_core/test_resolution_parity/test_parity_environment.py)
# ---------------------------------------------------------------------------

# Keeps exec'd classes alive across assertions, like IPython's Out[N] history. Leaking them for the
# process lifetime is safe: the autouse cleanup unbinds them from __main__, so they are no longer
# live-in-module and dedup preserves instead of re-raising, and they only match probe722i_redef.
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
    """Snapshot __main__ attrs and restore after each test (xdist safety, mirrors test_parity_environment)."""
    main_mod = sys.modules["__main__"]
    snapshot = set(main_mod.__dict__.keys())
    yield
    for key in set(main_mod.__dict__.keys()) - snapshot:
        main_mod.__dict__.pop(key, None)


# ---------------------------------------------------------------------------
# 1. diagnose success: recursive records, complete=True
# ---------------------------------------------------------------------------


def test_probe722i_diagnose_success_records_every_identification() -> None:
    """A two-feature request yields one RESOLVED record per identification in planning order."""
    report = mlodaAPI.diagnose(
        _success_features(),
        compute_frameworks={PythonDictFramework},
        plugin_collector=_success_collector(),
    )

    assert isinstance(report, ResolutionReport)
    assert report.complete is True
    assert isinstance(report.features, tuple)

    paths = [record.dependency_path for record in report.features]
    assert paths == [
        (BASE_FEATURE,),
        (DERIVED_FEATURE,),
        (DERIVED_FEATURE, BASE_FEATURE),
    ]
    assert all(record.outcome.status is ResolutionStatus.RESOLVED for record in report.features)

    by_path = {record.dependency_path: record for record in report.features}
    root_winner = by_path[(BASE_FEATURE,)].outcome.winner
    assert root_winner is not None
    assert root_winner.feature_group is SourceBase722I
    derived_winner = by_path[(DERIVED_FEATURE,)].outcome.winner
    assert derived_winner is not None
    assert derived_winner.feature_group is DerivedOnBase722I
    child_winner = by_path[(DERIVED_FEATURE, BASE_FEATURE)].outcome.winner
    assert child_winner is not None
    assert child_winner.feature_group is SourceBase722I

    # PINNED: a successful report still describes the environment it resolved against.
    assert report.environment is not None
    assert isinstance(report.environment, EnvironmentBuildOutcome)
    assert report.environment.errors == ()
    assert report.environment.snapshot is not None


def test_probe722i_diagnose_success_payload_is_plain_json() -> None:
    """to_payload() is json.dumps-able, delegates to member payloads, and leaks no class objects."""
    report = mlodaAPI.diagnose(
        _success_features(),
        compute_frameworks={PythonDictFramework},
        plugin_collector=_success_collector(),
    )

    payload = report.to_payload()
    text = json.dumps(payload)
    assert "<class" not in text

    assert payload["complete"] is True
    assert report.environment is not None
    assert payload["environment"] == report.environment.to_payload()
    features_payload = payload["features"]
    assert [entry["dependency_path"] for entry in features_payload] == [
        list(record.dependency_path) for record in report.features
    ]
    assert [entry["outcome"] for entry in features_payload] == [
        record.outcome.to_payload() for record in report.features
    ]


# ---------------------------------------------------------------------------
# 2. diagnose feature failure: non-raising, partial records retained
# ---------------------------------------------------------------------------


def test_probe722i_diagnose_feature_failure_is_non_raising_and_keeps_partial_records() -> None:
    """A feature matching nothing yields complete=False with earlier successes retained."""
    report = mlodaAPI.diagnose(
        [Feature(BASE_FEATURE), Feature(MISSING_FEATURE)],
        compute_frameworks={PythonDictFramework},
        plugin_collector=_success_collector(),
    )

    assert isinstance(report, ResolutionReport)
    assert report.complete is False
    assert len(report.features) == 2

    first, last = report.features
    assert first.dependency_path == (BASE_FEATURE,)
    assert first.outcome.status is ResolutionStatus.RESOLVED
    assert last.dependency_path == (MISSING_FEATURE,)
    assert last.outcome.status is ResolutionStatus.NOT_FOUND

    # The environment built fine; only the feature failed.
    assert report.environment is not None
    assert report.environment.snapshot is not None


# ---------------------------------------------------------------------------
# 3. diagnose environment failure: non-raising, environment errors carried
# ---------------------------------------------------------------------------


def test_probe722i_diagnose_environment_failure_is_non_raising() -> None:
    """A strict-mode empty-registry build yields complete=False with the strict-mode error, no features."""
    report = mlodaAPI.diagnose(
        [Feature(STRICT_FEATURE)],
        compute_frameworks={CfwStrict722I},
        plugin_collector=_strict_collector_with_empty_registry(),
    )

    assert isinstance(report, ResolutionReport)
    assert report.complete is False
    assert report.features == ()
    assert report.environment is not None
    assert report.environment.snapshot is None
    messages = [error.message for error in report.environment.errors]
    assert any(STRICT_MODE_TEXT in message for message in messages)


# ---------------------------------------------------------------------------
# 4. session.resolution_report(): planning-pass records, zero re-matching
# ---------------------------------------------------------------------------


def test_probe722i_session_resolution_report_reuses_planning_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolution_report() serves the engine's planning outcomes; calling it twice never re-resolves."""
    session = mlodaAPI.prepare(
        _success_features(),
        compute_frameworks={PythonDictFramework},
        plugin_collector=_success_collector(),
    )
    engine = session.engine
    assert engine is not None
    expected = tuple(
        FeatureResolutionRecord(dependency_path=path, outcome=outcome) for path, outcome in engine.resolution_outcomes
    )
    assert len(expected) == 3  # premise guard: root, derived, and the derived's input were identified

    calls = {"resolve": 0}
    original_resolve = FeatureGroupResolver.resolve

    def _counting_resolve(
        self: FeatureGroupResolver,
        request: ResolutionRequestSnapshot,
        environment: ResolutionEnvironmentSnapshot,
    ) -> ResolutionOutcome:
        calls["resolve"] += 1
        return original_resolve(self, request, environment)

    monkeypatch.setattr(FeatureGroupResolver, "resolve", _counting_resolve)

    first = session.resolution_report()
    second = session.resolution_report()

    assert calls["resolve"] == 0
    assert first.complete is True
    assert first.features == expected
    assert second.features == expected
    assert first.environment is not None
    assert first.environment.snapshot is not None


# ---------------------------------------------------------------------------
# 5. prepare() failure carries the partial report on the raised error
# ---------------------------------------------------------------------------


def test_probe722i_prepare_failure_error_carries_report() -> None:
    """The FeatureResolutionError raised through prepare() carries a partial ResolutionReport."""
    with pytest.raises(FeatureResolutionError) as exc_info:
        mlodaAPI.prepare(
            [Feature(MISSING_FEATURE)],
            compute_frameworks={PythonDictFramework},
            plugin_collector=_success_collector(),
        )

    error = exc_info.value
    assert NO_FEATURE_GROUPS_TEXT in str(error)  # message shape unchanged

    report = error.report
    assert isinstance(report, ResolutionReport)
    assert report.complete is False
    assert len(report.features) == 1
    record = report.features[0]
    assert record.dependency_path == (MISSING_FEATURE,)
    assert record.outcome.status is ResolutionStatus.NOT_FOUND
    assert record.outcome == error.outcome


# ---------------------------------------------------------------------------
# 6. Environment-failure raise-type preservation through the engine path
# ---------------------------------------------------------------------------


def test_probe722i_engine_path_strict_mode_raise_stays_value_error() -> None:
    """The engine-path strict-mode failure keeps satisfying pytest.raises(ValueError, match=...).

    INVESTIGATION RESULT: no existing test pins the exact exception type through Engine/mlodaAPI
    construction; today this path raises a plain ValueError (EnvironmentBuildError.as_exception()
    over the strict-mode message). pytest.raises(ValueError) also admits a FeatureResolutionError
    subclass, so the report-carrying assertion stays conditional per the Stage 4a brief.
    """
    with pytest.raises(ValueError, match=STRICT_MODE_TEXT) as exc_info:
        mlodaAPI.prepare(
            [Feature(STRICT_FEATURE)],
            compute_frameworks={CfwStrict722I},
            plugin_collector=_strict_collector_with_empty_registry(),
        )

    error = exc_info.value
    if isinstance(error, FeatureResolutionError):
        assert isinstance(error.report, ResolutionReport)
        assert error.report.complete is False


def test_probe722i_engine_path_redefinition_conflict_type_preserved() -> None:
    """A redefinition conflict through mlodaAPI construction keeps raising RedefinitionConflictError.

    INVESTIGATION RESULT: existing RedefinitionConflictError pins under tests/ all go through
    PreFilterPlugins or dedup_feature_group_subclasses directly; the engine path re-raises the
    carried original exception, so callers catching RedefinitionConflictError must keep working.
    No report assertion here: the raised type is not FeatureResolutionError.
    """
    v1, v2 = _make_conflicting_pair("ProbeRedefReport722I", REDEF_FEATURE)
    collector = PluginCollector.enabled_feature_groups({v1, v2})

    with pytest.raises(RedefinitionConflictError, match="redefined with different source code"):
        mlodaAPI.prepare(
            [Feature(REDEF_FEATURE)],
            compute_frameworks={PythonDictFramework},
            plugin_collector=collector,
        )


# ---------------------------------------------------------------------------
# 7. Determinism: identical diagnose() calls give payload-equal reports
# ---------------------------------------------------------------------------


def test_probe722i_diagnose_payloads_are_deterministic() -> None:
    """Two identical diagnose() calls serialize to the same canonical JSON."""
    first = mlodaAPI.diagnose(
        _success_features(),
        compute_frameworks={PythonDictFramework},
        plugin_collector=_success_collector(),
    )
    second = mlodaAPI.diagnose(
        _success_features(),
        compute_frameworks={PythonDictFramework},
        plugin_collector=_success_collector(),
    )

    assert json.dumps(first.to_payload(), sort_keys=True) == json.dumps(second.to_payload(), sort_keys=True)


# ---------------------------------------------------------------------------
# 8. Coexistence smoke: report and plan surfaces on one session, explain unchanged
# ---------------------------------------------------------------------------


def test_probe722i_resolution_report_coexists_with_plan_surfaces() -> None:
    """resolution_report() and resolved_plan() serve one session; explain() stays plan-based."""
    session = mlodaAPI.prepare(
        _success_features(),
        compute_frameworks={PythonDictFramework},
        plugin_collector=_success_collector(),
    )

    report = session.resolution_report()
    plan = session.resolved_plan()

    assert report.complete is True
    assert plan, "resolved_plan must keep returning a non-empty step list"
    assert all(isinstance(step, PlanStep) for step in plan)
    plan_names = {name for step in plan for name in step.feature_names}
    assert {BASE_FEATURE, DERIVED_FEATURE} <= plan_names

    explained = mlodaAPI.explain(
        _success_features(),
        compute_frameworks={PythonDictFramework},
        plugin_collector=_success_collector(),
    )
    explained_names = {name for step in explained for name in step.feature_names}
    assert {BASE_FEATURE, DERIVED_FEATURE} <= explained_names
