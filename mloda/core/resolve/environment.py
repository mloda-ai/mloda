from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import (
    PluginCollector,
    strict_mode_from_env,
)
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare import accessible_plugins as _prefilter
from mloda.core.prepare.accessible_plugins import (
    FeatureGroupEnvironmentMapping,
    PreFilterPlugins,
    RedefinitionConflictError,
    dedup_feature_group_subclasses,
    registry_for,
)
from mloda.core.resolve.identity import PluginIdentity


_STRICT_MODE_FEATURE_GROUPS_MESSAGE = (
    "Strict mode filtered out all FeatureGroups: none of the accessible FeatureGroups "
    "are registered in the plugin registry. Register them via mloda.user.register_plugin() or "
    "relax MLODA_PLUGIN_REGISTRY_STRICT to disable strict mode."
)

_STRICT_MODE_COMPUTE_FRAMEWORKS_MESSAGE = (
    "Strict mode filtered out all ComputeFrameworks: none of the requested ComputeFrameworks "
    "are registered in the plugin registry. Register them via mloda.user.register_plugin() or "
    "relax MLODA_PLUGIN_REGISTRY_STRICT to disable strict mode."
)

_NO_FEATURE_GROUPS_MESSAGE = "No feature groups are loaded. Did you call PluginLoader.all()?"


class EnvironmentProvenance(Enum):
    """Why a plugin ended up in, or out of, the resolution environment."""

    # Reserved for the Stage 3 full-universe candidate records.
    DISCOVERED = "discovered"
    DISABLED_BY_COLLECTOR = "disabled_by_collector"
    POLICY_REJECTED = "policy_rejected"
    UNAVAILABLE = "unavailable"
    NOT_ENABLED = "not_enabled"
    REDEFINITION_CONFLICT = "redefinition_conflict"
    INVALID_DECLARATION = "invalid_declaration"
    ACCESSIBLE = "accessible"


@dataclass(frozen=True)
class FrameworkRecord:
    """One declared compute framework of a feature group and its classification."""

    identity: PluginIdentity
    provenance: EnvironmentProvenance


@dataclass(frozen=True)
class FeatureGroupRecord:
    """One discovered feature group, its classification, and its framework records."""

    identity: PluginIdentity
    provenance: EnvironmentProvenance
    frameworks: tuple[FrameworkRecord, ...]


@dataclass(frozen=True)
class EnvironmentBuildError:
    """One fatal condition met while building the environment.

    ``exception`` carries the original provider or dedup exception (never serialized)
    so adapters can re-raise exactly what the absorbed inline logic raised.
    """

    category: str
    message: str
    conflict_identities: tuple[PluginIdentity, ...] = ()
    exception: Exception | None = None

    def as_exception(self) -> Exception:
        """Return the carried original exception, else a ValueError over the message."""
        if self.exception is not None:
            return self.exception
        return ValueError(self.message)

    def to_payload(self) -> dict[str, Any]:
        """Plain-data projection, redacted: raw provider text never appears.

        Errors carrying an exception name only its type; errors without one carry
        mloda-generated safe text, so their message stays.
        """
        payload: dict[str, Any] = {
            "category": self.category,
            "conflict_identities": [identity.render() for identity in self.conflict_identities],
        }
        if self.exception is not None:
            payload["exception_type"] = type(self.exception).__name__
        else:
            payload["message"] = self.message
        return payload


@dataclass(frozen=True)
class ResolutionEnvironmentSnapshot:
    """Immutable, deterministic view of the plugin environment for one build."""

    records: tuple[FeatureGroupRecord, ...]
    accessible: tuple[tuple[type[FeatureGroup], tuple[type[ComputeFramework], ...]], ...]
    strict_mode: str
    requested_frameworks: tuple[PluginIdentity, ...]
    enabled_frameworks: tuple[PluginIdentity, ...]
    fingerprint: str

    def accessible_mapping(self) -> FeatureGroupEnvironmentMapping:
        """Fresh mutable mapping per call, equal to PreFilterPlugins.get_accessible_plugins()."""
        return {feature_group: set(frameworks) for feature_group, frameworks in self.accessible}

    def to_payload(self) -> dict[str, Any]:
        """Plain JSON data; plugin classes are rendered as identity strings."""
        return {
            "strict_mode": self.strict_mode,
            "fingerprint": self.fingerprint,
            "requested_frameworks": [identity.render() for identity in self.requested_frameworks],
            "enabled_frameworks": [identity.render() for identity in self.enabled_frameworks],
            "records": [
                {
                    "identity": record.identity.render(),
                    "provenance": record.provenance.value,
                    "frameworks": [
                        {"identity": framework.identity.render(), "provenance": framework.provenance.value}
                        for framework in record.frameworks
                    ],
                }
                for record in self.records
            ],
            "accessible": [
                {
                    "feature_group": PluginIdentity.from_class(feature_group).render(),
                    "frameworks": [PluginIdentity.from_class(framework).render() for framework in frameworks],
                }
                for feature_group, frameworks in self.accessible
            ],
        }


@dataclass(frozen=True)
class EnvironmentBuildOutcome:
    """Snapshot on success, structured errors otherwise; never both."""

    snapshot: ResolutionEnvironmentSnapshot | None
    errors: tuple[EnvironmentBuildError, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """Plain JSON data for the whole outcome."""
        return {
            "snapshot": self.snapshot.to_payload() if self.snapshot is not None else None,
            "errors": [error.to_payload() for error in self.errors],
        }


@dataclass(frozen=True)
class FeatureGroupPipelineResult:
    """Survivors and per-class drops of the feature-group pipeline, or a fatal error."""

    survivors: tuple[type[FeatureGroup], ...]
    dropped: tuple[tuple[type[FeatureGroup], EnvironmentProvenance], ...]
    error: EnvironmentBuildError | None


@dataclass(frozen=True)
class ComputeFrameworkPipelineResult:
    """Enabled frameworks after intersection and strict filtering, or a fatal error.

    ``available`` is the single per-build availability sampling; classification reuses it.
    """

    enabled: frozenset[type[ComputeFramework]]
    error: EnvironmentBuildError | None
    available: frozenset[type[ComputeFramework]] = frozenset()


def _strict_mode(plugin_collector: PluginCollector | None) -> str:
    return plugin_collector.strict_mode if plugin_collector is not None else strict_mode_from_env()


def run_feature_group_pipeline(plugin_collector: PluginCollector | None = None) -> FeatureGroupPipelineResult:
    """Discover, collector-filter, strict-filter, and dedup feature groups.

    Mirrors the absorbed PreFilterPlugins._set_feature_groups step for step,
    returning structured drops and errors instead of raising.
    """
    discovered = set(PreFilterPlugins.get_featuregroup_subclasses())
    dropped: list[tuple[type[FeatureGroup], EnvironmentProvenance]] = []

    working = set(discovered)
    if plugin_collector is not None:
        for feature_group in sorted(discovered, key=PluginIdentity.from_class):
            if not plugin_collector.applicable_feature_group_class(feature_group):
                working.discard(feature_group)
                dropped.append((feature_group, EnvironmentProvenance.DISABLED_BY_COLLECTOR))

    allow_redefinition = plugin_collector.allow_redefinition if plugin_collector is not None else False
    strict_mode = _strict_mode(plugin_collector)

    registered: set[type[Any]] = set()
    if strict_mode != "off":
        registered = registry_for(plugin_collector).registered_classes()

    if strict_mode == "strict":
        before_strict = working
        working = {fg for fg in before_strict if fg in registered or inspect.isabstract(fg)}
        for feature_group in sorted(before_strict - working, key=PluginIdentity.from_class):
            dropped.append((feature_group, EnvironmentProvenance.POLICY_REJECTED))
        if plugin_collector is not None:
            dropped_enabled = sorted(
                f"{fg.__module__}:{fg.__qualname__}"
                for fg in before_strict - working
                if fg in plugin_collector.enabled_feature_group_classes
            )
            if dropped_enabled:
                _prefilter.logger.warning(
                    "Explicitly enabled FeatureGroups were dropped by strict mode because they are "
                    "not registered in the plugin registry: %s.",
                    ", ".join(dropped_enabled),
                )
        had_concrete = any(not inspect.isabstract(fg) for fg in before_strict)
        has_concrete = any(not inspect.isabstract(fg) for fg in working)
        if had_concrete and not has_concrete:
            error = EnvironmentBuildError(
                category="strict_mode_feature_groups", message=_STRICT_MODE_FEATURE_GROUPS_MESSAGE
            )
            return FeatureGroupPipelineResult(survivors=(), dropped=tuple(dropped), error=error)

    # Dedup raises on same-identity classes with differing source; the factory returns it structurally.
    try:
        working = dedup_feature_group_subclasses(working, allow_redefinition=allow_redefinition)
    except RedefinitionConflictError as exc:
        error = EnvironmentBuildError(
            category=EnvironmentProvenance.REDEFINITION_CONFLICT.value,
            message=str(exc),
            conflict_identities=tuple(sorted({PluginIdentity.from_class(cls) for cls in exc.conflicts})),
            exception=exc.with_traceback(None),
        )
        return FeatureGroupPipelineResult(survivors=(), dropped=tuple(dropped), error=error)

    if strict_mode == "warn":
        unregistered = sorted(
            f"{fg.__module__}:{fg.__qualname__}"
            for fg in working
            if fg not in registered
            and not inspect.isabstract(fg)
            and f"{fg.__module__}:{fg.__qualname__}" not in _prefilter._warned_unregistered
        )
        if unregistered:
            _prefilter._warned_unregistered.update(unregistered)
            _prefilter.logger.warning(
                "FeatureGroups not registered in the plugin registry: %s.",
                ", ".join(unregistered),
            )

    if len(working) == 0:
        error = EnvironmentBuildError(category="no_feature_groups", message=_NO_FEATURE_GROUPS_MESSAGE)
        return FeatureGroupPipelineResult(survivors=(), dropped=tuple(dropped), error=error)

    survivors = tuple(sorted(working, key=PluginIdentity.from_class))
    return FeatureGroupPipelineResult(survivors=survivors, dropped=tuple(dropped), error=None)


def run_compute_framework_pipeline(
    compute_frameworks: set[type[ComputeFramework]],
    plugin_collector: PluginCollector | None = None,
    available: frozenset[type[ComputeFramework]] | None = None,
) -> ComputeFrameworkPipelineResult:
    """Intersect the requested frameworks with the available ones and apply strict mode.

    Mirrors the absorbed PreFilterPlugins._set_compute_frameworks step for step.
    ``available`` reuses a caller-side availability sampling; None samples here.
    """
    if available is None:
        available = frozenset(PreFilterPlugins.get_cfw_subclasses())
    enabled = compute_frameworks.intersection(available)

    strict_mode = _strict_mode(plugin_collector)
    if strict_mode == "off":
        return ComputeFrameworkPipelineResult(enabled=frozenset(enabled), error=None, available=available)

    registered = registry_for(plugin_collector).registered_classes()
    # Bundled plugin frameworks ship with mloda, like abstract classes: never filtered or flagged.
    unregistered = {
        cfw
        for cfw in enabled
        if cfw not in registered and not inspect.isabstract(cfw) and not _prefilter._is_bundled_plugin(cfw)
    }
    unregistered_names = sorted(f"{cfw.__module__}:{cfw.__qualname__}" for cfw in unregistered)

    if strict_mode == "warn":
        new_names = [name for name in unregistered_names if name not in _prefilter._warned_unregistered]
        if new_names:
            _prefilter._warned_unregistered.update(new_names)
            _prefilter.logger.warning(
                "ComputeFrameworks not registered in the plugin registry: %s.", ", ".join(new_names)
            )
        return ComputeFrameworkPipelineResult(enabled=frozenset(enabled), error=None, available=available)

    surviving = enabled - unregistered
    had_concrete = any(not inspect.isabstract(cfw) for cfw in enabled)
    has_concrete = any(not inspect.isabstract(cfw) for cfw in surviving)
    if had_concrete and not has_concrete:
        error = EnvironmentBuildError(
            category="strict_mode_compute_frameworks", message=_STRICT_MODE_COMPUTE_FRAMEWORKS_MESSAGE
        )
        return ComputeFrameworkPipelineResult(enabled=frozenset(), error=error, available=available)
    return ComputeFrameworkPipelineResult(enabled=frozenset(surviving), error=None, available=available)


def _member_identity(member: Any) -> PluginIdentity:
    """Identity of a declared rule member, defensive: junk members may lack module/qualname."""
    module = getattr(member, "__module__", type(member).__module__)
    qualname = getattr(member, "__qualname__", type(member).__qualname__)
    return PluginIdentity(module=str(module), qualname=str(qualname))


def _fingerprint(
    strict_mode: str,
    requested: tuple[PluginIdentity, ...],
    enabled: tuple[PluginIdentity, ...],
    records: tuple[FeatureGroupRecord, ...],
) -> str:
    """sha256 hex over a canonical string of identities, provenance, and run configuration."""
    canonical = json.dumps(
        {
            "strict_mode": strict_mode,
            "requested_frameworks": [identity.render() for identity in requested],
            "enabled_frameworks": [identity.render() for identity in enabled],
            "records": [
                [
                    record.identity.render(),
                    record.provenance.value,
                    [[framework.identity.render(), framework.provenance.value] for framework in record.frameworks],
                ]
                for record in records
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_resolution_environment(
    compute_frameworks: set[type[ComputeFramework]] | None = None,
    plugin_collector: PluginCollector | None = None,
) -> EnvironmentBuildOutcome:
    """Run the PreFilterPlugins classification pipeline, returning structured state.

    Same evaluation order as PreFilterPlugins; on the conditions where it raises,
    the outcome carries an error with the exact current message and no snapshot.
    Availability is sampled exactly once inside this pipeline for both the
    ``compute_frameworks=None`` ("all available frameworks") and the explicit-set
    branch, and a raising ``is_available`` probe becomes a structured
    availability_failure error instead of a raise.
    """
    # Guarded: availability probes are third-party code; a raise must fail structurally here.
    try:
        sampled = frozenset(PreFilterPlugins.get_cfw_subclasses())
    except Exception as exc:
        error = EnvironmentBuildError(
            category="availability_failure",
            message=str(exc),
            exception=exc.with_traceback(None),
        )
        return EnvironmentBuildOutcome(snapshot=None, errors=(error,))
    if compute_frameworks is None:
        compute_frameworks = set(sampled)

    strict_mode = _strict_mode(plugin_collector)
    requested = tuple(sorted(PluginIdentity.from_class(cfw) for cfw in compute_frameworks))

    feature_group_result = run_feature_group_pipeline(plugin_collector)
    if feature_group_result.error is not None:
        return EnvironmentBuildOutcome(snapshot=None, errors=(feature_group_result.error,))

    framework_result = run_compute_framework_pipeline(compute_frameworks, plugin_collector, available=sampled)
    if framework_result.error is not None:
        return EnvironmentBuildOutcome(snapshot=None, errors=(framework_result.error,))

    enabled_classes = framework_result.enabled
    # Classification reuses the pipeline's one availability sampling; is_available never runs again.
    available_classes = framework_result.available
    enabled = tuple(sorted(PluginIdentity.from_class(cfw) for cfw in enabled_classes))

    records_by_identity: dict[PluginIdentity, FeatureGroupRecord] = {}
    for feature_group, provenance in feature_group_result.dropped:
        identity = PluginIdentity.from_class(feature_group)
        records_by_identity[identity] = FeatureGroupRecord(identity=identity, provenance=provenance, frameworks=())

    accessible_entries: list[tuple[type[FeatureGroup], tuple[type[ComputeFramework], ...]]] = []
    for feature_group in feature_group_result.survivors:
        identity = PluginIdentity.from_class(feature_group)
        # Provider declarations are third-party code; a raise becomes a structured invalid_declaration error.
        try:
            declared = feature_group.compute_framework_definition()
        except Exception as exc:
            error = EnvironmentBuildError(
                category=EnvironmentProvenance.INVALID_DECLARATION.value,
                message=str(exc),
                exception=exc.with_traceback(None),
            )
            return EnvironmentBuildOutcome(snapshot=None, errors=(error,))

        framework_records: list[FrameworkRecord] = []
        accessible_frameworks: list[type[ComputeFramework]] = []
        # The declared type lies for junk members (anything a provider put in the rule set).
        members: list[Any] = sorted(declared, key=_member_identity)
        for member in members:
            identity_record = _member_identity(member)
            if not (isinstance(member, type) and issubclass(member, ComputeFramework)):
                # Junk members are recorded, never fatal; PreFilterPlugins keeps dropping them silently.
                framework_records.append(
                    FrameworkRecord(identity=identity_record, provenance=EnvironmentProvenance.INVALID_DECLARATION)
                )
                continue
            framework: type[ComputeFramework] = member
            if framework not in available_classes:
                framework_provenance = EnvironmentProvenance.UNAVAILABLE
            elif framework in enabled_classes:
                framework_provenance = EnvironmentProvenance.ACCESSIBLE
                accessible_frameworks.append(framework)
            elif framework in compute_frameworks:
                framework_provenance = EnvironmentProvenance.POLICY_REJECTED
            else:
                framework_provenance = EnvironmentProvenance.NOT_ENABLED
            framework_records.append(FrameworkRecord(identity=identity_record, provenance=framework_provenance))

        records_by_identity[identity] = FeatureGroupRecord(
            identity=identity, provenance=EnvironmentProvenance.ACCESSIBLE, frameworks=tuple(framework_records)
        )
        accessible_entries.append((feature_group, tuple(accessible_frameworks)))

    records = tuple(records_by_identity[identity] for identity in sorted(records_by_identity))
    snapshot = ResolutionEnvironmentSnapshot(
        records=records,
        accessible=tuple(accessible_entries),
        strict_mode=strict_mode,
        requested_frameworks=requested,
        enabled_frameworks=enabled,
        fingerprint=_fingerprint(strict_mode, requested, enabled, records),
    )
    return EnvironmentBuildOutcome(snapshot=snapshot, errors=())
