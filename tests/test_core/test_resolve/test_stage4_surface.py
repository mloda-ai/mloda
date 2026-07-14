"""Target-contract tests for the final Stage 4 surface of issue #722 (Stage 4b).

Stage 4b completes the resolve_feature request surface and the persona renderers:

- resolve_feature accepts ``str | Feature`` as its first argument plus the keyword-only
  parameters ``links``, ``data_access_collection``, and ``compute_frameworks``. A Feature
  object carries its own options, domain, scope, and framework pin; combining a Feature
  with the ``options`` or ``feature_group`` keyword returns an error ResolvedFeature
  (never a raise). ``compute_frameworks=None`` keeps the all-available standalone default;
  a set restricts the run set exactly like ``mlodaAPI(compute_frameworks=...)``.
- ResolvedFeature carries ``mode == "standalone"``, the explicit diagnostics-mode label.
- ``mloda.core.resolve.render`` exposes pure persona renderers over a captured
  ResolutionOutcome: render_user_view (str), render_provider_view (str),
  render_steward_view (JSON-serializable dict). Renderers NEVER invoke provider hooks.
- ``mloda/core/api/plugin_docs.py`` structurally stops referencing the matching hooks
  (match_feature_group_criteria, supports_compute_framework, split_frameworks_by_capability);
  the redefinition-branch candidate filtering moves into the resolve layer, behavior preserved.

Every test except the redefinition-conflict pin FAILS until the Green phase lands.
Probe names use the unique probe722j_ prefix so process-global scans in other tests
never trip these fixtures.
"""

from __future__ import annotations

import inspect
import json
import linecache
import sys
import textwrap
from typing import Any, ClassVar, Iterator, cast

import pytest

import mloda.core.api.plugin_docs as plugin_docs_module
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.api.plugin_info import ResolvedFeature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.resolve.identity import PluginIdentity
from mloda.core.resolve.outcome import FrameworkStatus, ResolutionOutcome, ResolutionStatus
from mloda.core.resolve.request import ResolutionRequestSnapshot
from mloda.core.resolve.resolver import FeatureGroupResolver, snapshot_from_mapping


DOMAIN_FEATURE = "probe722j_domain"
DOMAIN_A = "probe722j_domain_a"
INDEXED_FEATURE = "probe722j_indexed"
PIN_FEATURE = "probe722j_pin"
DAC_FEATURE = "probe722j_dac"
TWO_FRAMEWORK_FEATURE = "probe722j_twofw"
MODE_FEATURE = "probe722j_mode"
MODE_MISSING_FEATURE = "probe722j_mode_missing"
RENDER_WIN_FEATURE = "probe722j_render_win"
RENDER_AMBIGUOUS_FEATURE = "probe722j_render_ambiguous"
RENDER_FAIL_FEATURE = "probe722j_render_fail"
REDEF_FEATURE = "probe722j_redef"
REDEF_OTHER_NAME = "probe722j_redef_other"

STEWARD_SECRET_TEXT = "steward secret 722j"  # nosec B105
CONFLICT_TEXT = "cannot combine"
REDEFINITION_TEXT = "redefined with different source code"


# ---------------------------------------------------------------------------
# Mock compute frameworks (unique names, suffix 722J)
# ---------------------------------------------------------------------------


class CfwSurface722J(ComputeFramework):
    """General-purpose framework for the request-surface probes."""


class CfwPinA722J(ComputeFramework):
    """Uniquely named framework, pinnable by name from a Feature object."""


class CfwPinB722J(ComputeFramework):
    """Rival uniquely named framework for the sibling probe."""


class CfwOnly722J(ComputeFramework):
    """The framework the restricted run set keeps."""


class CfwOther722J(ComputeFramework):
    """The framework the restricted run set excludes."""


class CfwRender722J(ComputeFramework):
    """Framework carrying the renderer probes."""


# ---------------------------------------------------------------------------
# Probe feature groups (match ONLY their own probe722j_* names)
# ---------------------------------------------------------------------------


class _SurfaceProbeBase722J(FeatureGroup):
    """Shared probe base: never matches anything itself."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwSurface722J}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class ProbeDomainA722J(_SurfaceProbeBase722J):
    """Matches the shared domain feature name; lives in domain A."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain(DOMAIN_A)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == DOMAIN_FEATURE


class ProbeDomainB722J(_SurfaceProbeBase722J):
    """Matches the shared domain feature name; lives in domain B."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("probe722j_domain_b")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == DOMAIN_FEATURE


class ProbeLinkLeft722J(_SurfaceProbeBase722J):
    """Anchors the left side of the foreign link; never matches a name."""


class ProbeLinkRight722J(_SurfaceProbeBase722J):
    """Anchors the right side of the foreign link; never matches a name."""


class ProbeIndexed722J(_SurfaceProbeBase722J):
    """Declares an index no link in the request carries."""

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("probe722j_row_id",))]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == INDEXED_FEATURE


class ProbePinA722J(_SurfaceProbeBase722J):
    """Sibling probe on CfwPinA722J, matching the shared pin feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwPinA722J}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == PIN_FEATURE


class ProbePinB722J(_SurfaceProbeBase722J):
    """Sibling probe on CfwPinB722J, matching the same pin feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwPinB722J}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == PIN_FEATURE


class ProbeDac722J(_SurfaceProbeBase722J):
    """Matches its name only when a data access collection is present."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == DAC_FEATURE and data_access_collection is not None


class ProbeTwoFw722J(_SurfaceProbeBase722J):
    """Declares two available frameworks; the restricted run set enables only one."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwOnly722J, CfwOther722J}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == TWO_FRAMEWORK_FEATURE


class ProbeMode722J(_SurfaceProbeBase722J):
    """Plain clean probe for the mode-label and conflict tests."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == MODE_FEATURE


# ---------------------------------------------------------------------------
# Counting renderer probes: every criteria/capability hook call bumps the counters,
# so the purity test can prove the renderers never invoke provider hooks.
# ---------------------------------------------------------------------------

_RENDER_HOOK_CALLS: dict[str, int] = {"criteria": 0, "capability": 0}


class _RenderProbeBase722J(FeatureGroup):
    """Counting probe base: matches only its own RENDER_NAME."""

    RENDER_NAME: ClassVar[str] = ""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwRender722J}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        _RENDER_HOOK_CALLS["criteria"] += 1
        return str(feature_name) == cls.RENDER_NAME

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        _RENDER_HOOK_CALLS["capability"] += 1
        return True

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class RenderWinner722J(_RenderProbeBase722J):
    """Sole candidate of the RESOLVED renderer outcome."""

    RENDER_NAME: ClassVar[str] = RENDER_WIN_FEATURE


class RenderRivalA722J(_RenderProbeBase722J):
    """First survivor of the AMBIGUOUS renderer outcome."""

    RENDER_NAME: ClassVar[str] = RENDER_AMBIGUOUS_FEATURE


class RenderRivalB722J(_RenderProbeBase722J):
    """Second survivor of the AMBIGUOUS renderer outcome."""

    RENDER_NAME: ClassVar[str] = RENDER_AMBIGUOUS_FEATURE


class RenderBoomCap722J(_RenderProbeBase722J):
    """Capability hook raises, producing the FAILED renderer outcome; gated on its name."""

    RENDER_NAME: ClassVar[str] = RENDER_FAIL_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        _RENDER_HOOK_CALLS["capability"] += 1
        if str(feature_name) == RENDER_FAIL_FEATURE:
            raise ValueError(STEWARD_SECRET_TEXT)
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _foreign_link() -> Link:
    """A link between two other groups whose indexes the indexed probe does not support."""
    return Link.inner(
        JoinSpec(ProbeLinkLeft722J, "probe722j_left_id"),
        JoinSpec(ProbeLinkRight722J, "probe722j_right_id"),
    )


def _render_outcome(feature_name: str, mapping: FeatureGroupEnvironmentMapping) -> ResolutionOutcome:
    """One captured outcome over a hand-built mapping, built by the authoritative resolver."""
    request = ResolutionRequestSnapshot.from_feature(Feature(feature_name))
    return FeatureGroupResolver().resolve(request, snapshot_from_mapping(mapping))


def _resolved_outcome() -> ResolutionOutcome:
    outcome = _render_outcome(RENDER_WIN_FEATURE, {RenderWinner722J: {CfwRender722J}})
    assert outcome.status is ResolutionStatus.RESOLVED  # premise guard
    return outcome


def _ambiguous_outcome() -> ResolutionOutcome:
    outcome = _render_outcome(
        RENDER_AMBIGUOUS_FEATURE,
        {RenderRivalA722J: {CfwRender722J}, RenderRivalB722J: {CfwRender722J}},
    )
    assert outcome.status is ResolutionStatus.AMBIGUOUS  # premise guard
    return outcome


def _failed_outcome() -> ResolutionOutcome:
    outcome = _render_outcome(RENDER_FAIL_FEATURE, {RenderBoomCap722J: {CfwRender722J}})
    assert outcome.status is ResolutionStatus.FAILED  # premise guard
    return outcome


# ---------------------------------------------------------------------------
# Jupyter-cell simulation helpers for the redefinition-conflict pin
# (construction pattern mirrors tests/test_core/test_resolution_parity/test_parity_environment.py)
# ---------------------------------------------------------------------------

# Keeps exec'd classes alive across assertions, like IPython's Out[N] history. Leaking them for the process
# lifetime is safe: the autouse cleanup unbinds them from __main__, so they are no longer live-in-module and
# dedup preserves instead of re-raising, and they only match the unique name probe722j_redef.
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
# 1. Feature-object expressibility parity: domain
# ---------------------------------------------------------------------------


def test_probe722j_feature_object_domain_matches_engine_verdict() -> None:
    """A Feature object carries its domain into resolve_feature; verdict matches the engine."""
    mapping: FeatureGroupEnvironmentMapping = {
        ProbeDomainA722J: {CfwSurface722J},
        ProbeDomainB722J: {CfwSurface722J},
    }
    identifier = IdentifyFeatureGroupClass(
        feature=Feature(DOMAIN_FEATURE, domain=DOMAIN_A),
        accessible_plugins=mapping,
        links=None,
    )
    engine_resolved, _ = identifier.get()
    assert engine_resolved is ProbeDomainA722J  # premise guard: the engine verdict

    collector = PluginCollector.enabled_feature_groups({ProbeDomainA722J, ProbeDomainB722J})
    result = resolve_feature(Feature(DOMAIN_FEATURE, domain=DOMAIN_A), plugin_collector=collector)

    # TARGET CONTRACT: the Feature's domain is expressible, resolving the domain-A probe
    # exactly like the engine.
    assert result.feature_group is engine_resolved
    assert result.error is None


# ---------------------------------------------------------------------------
# 2. Links parity
# ---------------------------------------------------------------------------


def test_probe722j_links_parameter_excludes_unsupported_index_probe() -> None:
    """The links parameter carries the run's links; the indexed probe stops resolving."""
    collector = PluginCollector.enabled_feature_groups({ProbeIndexed722J})

    # Premise guard: without links the indexed probe resolves cleanly.
    unlinked = resolve_feature(INDEXED_FEATURE, plugin_collector=collector)
    assert unlinked.feature_group is ProbeIndexed722J
    assert unlinked.error is None

    result = resolve_feature(INDEXED_FEATURE, links={_foreign_link()}, plugin_collector=collector)

    # TARGET CONTRACT: the probe whose index no request link carries is reported as not
    # resolvable, exactly like the engine's links filter.
    assert result.feature_group is None
    assert result.error is not None


# ---------------------------------------------------------------------------
# 3. Pin parity
# ---------------------------------------------------------------------------


def test_probe722j_feature_object_pin_disambiguates_framework_siblings() -> None:
    """A framework pin on the Feature object resolves the per-framework sibling ambiguity."""
    collector = PluginCollector.enabled_feature_groups({ProbePinA722J, ProbePinB722J})

    # Premise guard: unpinned, the framework siblings stay ambiguous.
    unpinned = resolve_feature(PIN_FEATURE, plugin_collector=collector)
    assert unpinned.feature_group is None
    assert unpinned.error is not None
    assert "Multiple FeatureGroups match" in unpinned.error

    result = resolve_feature(
        Feature(PIN_FEATURE, compute_framework=CfwPinA722J.get_class_name()), plugin_collector=collector
    )

    # TARGET CONTRACT: the pin resolves the pinned sibling with exactly its framework.
    assert result.feature_group is ProbePinA722J
    assert result.error is None
    assert result.supported_compute_frameworks == [CfwPinA722J.get_class_name()]


# ---------------------------------------------------------------------------
# 4. Data access collection parity
# ---------------------------------------------------------------------------


def test_probe722j_data_access_collection_parameter_enables_reader_probe() -> None:
    """The data_access_collection parameter reaches matching, exactly like the engine threads it."""
    collector = PluginCollector.enabled_feature_groups({ProbeDac722J})

    # Premise guard: without a collection the reader probe does not match.
    without = resolve_feature(DAC_FEATURE, plugin_collector=collector)
    assert without.feature_group is None
    assert without.error is not None
    assert "No FeatureGroup found" in without.error

    result = resolve_feature(
        DAC_FEATURE,
        data_access_collection=DataAccessCollection(files={"probe722j": "probe722j.csv"}),
        plugin_collector=collector,
    )

    # TARGET CONTRACT: with the collection the reader probe resolves.
    assert result.feature_group is ProbeDac722J
    assert result.error is None


# ---------------------------------------------------------------------------
# 5. Run-set parity
# ---------------------------------------------------------------------------


def test_probe722j_compute_frameworks_parameter_restricts_run_set() -> None:
    """The compute_frameworks parameter restricts the standalone run set to the given frameworks."""
    collector = PluginCollector.enabled_feature_groups({ProbeTwoFw722J})

    # Premise guard: the standalone default reports every installed declared framework.
    unrestricted = resolve_feature(TWO_FRAMEWORK_FEATURE, plugin_collector=collector)
    assert unrestricted.feature_group is ProbeTwoFw722J
    assert unrestricted.supported_compute_frameworks == [
        CfwOnly722J.get_class_name(),
        CfwOther722J.get_class_name(),
    ]

    result = resolve_feature(TWO_FRAMEWORK_FEATURE, compute_frameworks={CfwOnly722J}, plugin_collector=collector)

    # TARGET CONTRACT: compute_frameworks restricts the standalone environment exactly like
    # mlodaAPI(compute_frameworks=...) restricts the engine's run set.
    assert result.feature_group is ProbeTwoFw722J
    assert result.error is None
    assert result.supported_compute_frameworks == [CfwOnly722J.get_class_name()]
    assert result.unsupported_compute_frameworks == []


# ---------------------------------------------------------------------------
# 6. Feature object combined with the options or feature_group keyword
# ---------------------------------------------------------------------------


def test_probe722j_feature_object_with_options_keyword_is_conflict_error() -> None:
    """A Feature carries its own options; adding the options keyword is an error result, never a raise."""
    collector = PluginCollector.enabled_feature_groups({ProbeMode722J})

    result = resolve_feature(
        Feature(MODE_FEATURE),
        options=Options(group={"probe722j_extra": 1}),
        plugin_collector=collector,
    )

    # TARGET CONTRACT: the conflict is reported in the non-throwing debug shape.
    assert isinstance(result, ResolvedFeature)
    assert result.feature_group is None
    assert result.error is not None
    assert CONFLICT_TEXT in result.error.lower()


def test_probe722j_feature_object_with_feature_group_keyword_is_conflict_error() -> None:
    """A Feature carries its own scope; adding the feature_group keyword is an error result, never a raise."""
    collector = PluginCollector.enabled_feature_groups({ProbeMode722J})

    result = resolve_feature(Feature(MODE_FEATURE), feature_group=ProbeMode722J, plugin_collector=collector)

    # TARGET CONTRACT: the conflict is reported in the non-throwing debug shape.
    assert isinstance(result, ResolvedFeature)
    assert result.feature_group is None
    assert result.error is not None
    assert CONFLICT_TEXT in result.error.lower()


# ---------------------------------------------------------------------------
# 7. Diagnostics-mode label
# ---------------------------------------------------------------------------


def test_probe722j_resolved_feature_carries_standalone_mode_label() -> None:
    """Every ResolvedFeature is labelled with the standalone diagnostics mode."""
    # TARGET CONTRACT: the field defaults to the standalone label.
    bare = ResolvedFeature(feature_name="probe722j_bare", feature_group=None, candidates=[], error=None)
    assert bare.mode == "standalone"

    collector = PluginCollector.enabled_feature_groups({ProbeMode722J})

    success = resolve_feature(MODE_FEATURE, plugin_collector=collector)
    assert success.feature_group is ProbeMode722J
    assert success.mode == "standalone"

    missing = resolve_feature(MODE_MISSING_FEATURE, plugin_collector=collector)
    assert missing.feature_group is None
    assert missing.error is not None
    assert missing.mode == "standalone"


# ---------------------------------------------------------------------------
# 8. Persona renderers over captured outcomes
# ---------------------------------------------------------------------------


def test_probe722j_render_user_view_names_status_and_winner_or_failure() -> None:
    """The user view mentions the status and the winner or the failing plugin."""
    from mloda.core.resolve.render import render_user_view

    resolved_text = render_user_view(_resolved_outcome())
    assert ResolutionStatus.RESOLVED.value in resolved_text.lower()
    assert "RenderWinner722J" in resolved_text

    ambiguous_text = render_user_view(_ambiguous_outcome())
    assert ResolutionStatus.AMBIGUOUS.value in ambiguous_text.lower()

    failed_text = render_user_view(_failed_outcome())
    assert ResolutionStatus.FAILED.value in failed_text.lower()
    assert "RenderBoomCap722J" in failed_text


def test_probe722j_render_provider_view_lists_candidates_and_framework_statuses() -> None:
    """The provider view carries every candidate identity render and the framework status values."""
    from mloda.core.resolve.render import render_provider_view

    resolved_text = render_provider_view(_resolved_outcome())
    assert PluginIdentity.from_class(RenderWinner722J).render() in resolved_text
    assert FrameworkStatus.SUPPORTED.value in resolved_text.lower()

    ambiguous_text = render_provider_view(_ambiguous_outcome())
    assert PluginIdentity.from_class(RenderRivalA722J).render() in ambiguous_text
    assert PluginIdentity.from_class(RenderRivalB722J).render() in ambiguous_text
    assert FrameworkStatus.SUPPORTED.value in ambiguous_text.lower()

    failed_text = render_provider_view(_failed_outcome())
    assert PluginIdentity.from_class(RenderBoomCap722J).render() in failed_text
    assert FrameworkStatus.HOOK_FAILED.value in failed_text.lower()


def test_probe722j_render_steward_view_is_deterministic_serializable_and_redacted() -> None:
    """The steward view is a deterministic JSON dict carrying the fingerprint, never raw provider text."""
    from mloda.core.resolve.render import render_steward_view

    resolved = _resolved_outcome()
    view = render_steward_view(resolved)
    assert isinstance(view, dict)
    first = json.dumps(view, sort_keys=True)
    second = json.dumps(render_steward_view(resolved), sort_keys=True)
    assert first == second
    assert resolved.environment_fingerprint in first
    assert PluginIdentity.from_class(RenderWinner722J).render() in first
    assert ResolutionStatus.RESOLVED.value in first

    failed = _failed_outcome()
    failed_dump = json.dumps(render_steward_view(failed), sort_keys=True)
    assert failed.environment_fingerprint in failed_dump
    assert "ValueError" in failed_dump  # the failure category stays visible
    assert STEWARD_SECRET_TEXT not in failed_dump  # the raw provider message is redacted


def test_probe722j_renderers_never_invoke_provider_hooks() -> None:
    """All three renderers are pure projections: zero criteria/capability hook invocations."""
    from mloda.core.resolve.render import render_provider_view, render_steward_view, render_user_view

    baseline = dict(_RENDER_HOOK_CALLS)
    outcomes = [_resolved_outcome(), _ambiguous_outcome(), _failed_outcome()]
    after_build = dict(_RENDER_HOOK_CALLS)
    # Premise guard: the single build pass DOES invoke the hooks; only rendering must not.
    assert after_build["criteria"] > baseline["criteria"]
    assert after_build["capability"] > baseline["capability"]

    for outcome in outcomes:
        render_user_view(outcome)
        render_provider_view(outcome)
        render_steward_view(outcome)

    assert dict(_RENDER_HOOK_CALLS) == after_build


# ---------------------------------------------------------------------------
# 9. Structural guarantee: plugin_docs never references the matching hooks
# ---------------------------------------------------------------------------


def test_probe722j_plugin_docs_source_carries_no_matching_hook_references() -> None:
    """plugin_docs.py delegates all matching to the resolve layer; no hook name appears in its source."""
    source = inspect.getsource(plugin_docs_module)

    assert "match_feature_group_criteria" not in source
    assert "supports_compute_framework" not in source
    assert "split_frameworks_by_capability" not in source


# ---------------------------------------------------------------------------
# 10. Redefinition-conflict projection preserved after the move (may pass today)
# ---------------------------------------------------------------------------


def test_probe722j_redefinition_conflict_error_keeps_scope_and_name_matched_candidates() -> None:
    """TARGET CONTRACT pin: the conflict projection survives the move out of plugin_docs.

    The redefinition-branch candidate filtering moves into the resolve layer in Stage 4;
    this pin keeps the observable behavior fixed: the conflict error text is returned with
    exactly the scope- and name-matched conflicting candidates.
    """
    v1, v2 = _make_conflicting_pair("ProbeRedefSurface722J", REDEF_FEATURE)
    collector = PluginCollector.enabled_feature_groups({v1, v2})

    result = resolve_feature(REDEF_FEATURE, plugin_collector=collector)
    assert result.feature_group is None
    assert result.error is not None
    assert REDEFINITION_TEXT in result.error
    assert set(result.candidates) == {v1, v2}

    # Name filter: a non-matching feature name keeps the conflict error but drops the candidates.
    other = resolve_feature(REDEF_OTHER_NAME, plugin_collector=collector)
    assert other.feature_group is None
    assert other.error is not None
    assert REDEFINITION_TEXT in other.error
    assert other.candidates == []

    # Scope filter: a scope matching neither conflicting class drops the candidates too.
    scoped = resolve_feature(REDEF_FEATURE, feature_group="ProbeSomewhereElse722J", plugin_collector=collector)
    assert scoped.feature_group is None
    assert scoped.error is not None
    assert REDEFINITION_TEXT in scoped.error
    assert scoped.candidates == []
