"""Projection of a failed EvaluationResult into its message; imports the types, never the matcher."""

from difflib import get_close_matches

from mloda.core.abstract_plugins.components.utils import safe_field, as_str
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.prepare.resolution_types import EliminationStage, EvaluationResult


TROUBLESHOOTING_URL = "https://mloda-ai.github.io/mloda/in_depth/troubleshooting/feature-group-resolution-errors/"

MAX_SUGGESTIONS = 5


def scope_callout(scope: str | type[FeatureGroup] | None) -> str | None:
    """Render the shared scope callout, or None when the scope is unset."""
    if scope is None:
        return None
    scope_name = scope.get_class_name() if isinstance(scope, type) else scope
    return f"Scoped to feature group: '{scope_name}'."


def _candidate_sort_key(feature_group: type[FeatureGroup]) -> tuple[str, str]:
    """Sort candidates by name, then module: two candidates may share a name across modules."""
    return feature_group.__name__, feature_group.__module__


def _supported_feature_names(feature_group: type[FeatureGroup]) -> set[str]:
    """Best-effort name catalog of one candidate. A malformed entry costs that candidate its whole catalog."""
    return safe_field(
        lambda: {as_str(name) for name in feature_group.feature_names_supported()},
        set(),
        field=f"{feature_group.get_class_name()}.feature_names_supported",
    )


def _prefix_name(feature_group: type[FeatureGroup]) -> str:
    """Best-effort prefix of one candidate."""
    return safe_field(lambda: as_str(feature_group.prefix()), "", field=f"{feature_group.get_class_name()}.prefix")


_STAGE_LABELS: dict[EliminationStage, str] = {
    "value_rejection": "option value",
    "matcher_error": "match hook",
    "domain": "domain",
    "scope": "scope",
    "capability": "compute framework",
    "frameworks_not_enabled": "compute framework",
    "framework_pin": "compute framework pin",
    "links": "links",
}


def _render_near_miss_block(result: EvaluationResult, feature: Feature) -> str | None:
    """Shared section naming each eliminated near-miss candidate, its gate label, and its reason."""
    if not result.eliminations:
        return None
    lines = "\n".join(
        f"  - {fg.__name__} ({_STAGE_LABELS[result.eliminations[fg].stage]}): {result.eliminations[fg].reason}"
        for fg in sorted(result.eliminations, key=_candidate_sort_key)
    )
    return f"Feature group(s) eliminated while matching '{str(feature.name)}':\n{lines}"


def _render_multiple(result: EvaluationResult, feature: Feature, callout: str | None) -> str:
    # Every identified candidate gets a line; only a candidate with a captured domain gets the suffix.
    lines = "\n".join(
        f"  - {fg.__name__} ({fg.__module__})"
        + (f" [domain: {result.facts.domains[fg]}]" if fg in result.facts.domains else "")
        for fg in sorted(result.identified, key=_candidate_sort_key)
    )
    scope_line = f"{callout}\n" if callout else ""
    return (
        f"Multiple feature groups found for feature '{str(feature.name)}':\n"
        f"{lines}\n"
        f"{scope_line}"
        f"For troubleshooting guide, see: {TROUBLESHOOTING_URL}"
    )


def _render_abstract_only(result: EvaluationResult, feature: Feature, callout: str | None) -> str:
    feature_name = str(feature.name)
    if not result.facts.concrete_frameworks:
        msg = (
            f"No feature groups found for feature name: '{feature_name}'. "
            f"Only abstract feature group base(s) matched, which cannot be instantiated; "
            f"no concrete implementation is available or enabled."
        )
    else:
        framework_names = sorted(result.facts.concrete_frameworks)
        msg = (
            f"No feature groups found for feature name: '{feature_name}'. "
            f"Its concrete implementations require compute framework(s) {framework_names}, "
            f"none of which are available or enabled for this run."
        )

    # Place the callout on the sentence line, before the near-miss block, exactly as _render_none does, so it
    # is never space-glued to the last near-miss bullet.
    if callout:
        msg += f" {callout}"

    near_miss = _render_near_miss_block(result, feature)
    if near_miss is not None:
        msg += f"\n{near_miss}"
    return msg


def _render_none(result: EvaluationResult, feature: Feature, callout: str | None) -> str:
    feature_name = str(feature.name)
    msg = f"No feature groups found for feature name: '{feature_name}'."

    if callout:
        msg += f" {callout}"

    near_miss = _render_near_miss_block(result, feature)
    if near_miss is not None:
        msg += f"\n{near_miss}"

    # A suggestion equal to the requested name, echoing an already-named candidate, or reaching only groups this
    # pass killed, carries nothing new. Drop it, and the catalog's repeats, before the cut, so none spends a slot.
    droppable = {feature_name, *result.facts.eliminated_hints, *result.facts.dead_only_names}
    known_names = [name for name in dict.fromkeys(result.facts.known_names) if name not in droppable]
    similar = get_close_matches(feature_name, known_names, n=MAX_SUGGESTIONS, cutoff=0.5)
    if similar:
        msg += f"\nDid you mean one of: {similar}?"

    pointer_args = "name, options=..., feature_group=..." if callout else "name, options=..."
    msg += (
        f"\nUse resolve_feature({pointer_args}) to debug feature resolution."
        f"\nFor troubleshooting guide, see: {TROUBLESHOOTING_URL}"
    )
    return msg


def render_resolution_failure(result: EvaluationResult, feature: Feature) -> str | None:
    """Project a failed EvaluationResult into its message. Pure: reads only the result and the Feature.

    Calls no provider-overridable hook, so every fact it needs was captured by evaluate(). The
    forwarding hint is dropped: it needs a speculative second match, which is not a projection.
    """
    kind = result.failure_kind
    if kind is None:
        return None

    callout = scope_callout(feature.feature_group_scope)

    if kind == "multiple":
        return _render_multiple(result, feature, callout)

    if result.abstract_matched:
        return _render_abstract_only(result, feature, callout)

    return _render_none(result, feature, callout)
