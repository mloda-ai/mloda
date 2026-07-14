"""Persona renderers over a captured ResolutionOutcome.

Pure projections: they never invoke provider hooks, never build environments,
and never import from mloda.core.api.
"""

from __future__ import annotations

from typing import Any

from mloda.core.resolve.outcome import CandidateStatus, ResolutionOutcome, ResolutionStatus


def render_user_view(outcome: ResolutionOutcome) -> str:
    """Concise user summary: the status, the winner or failing plugin, and one next step."""
    lines = [f"Status: {outcome.status.value}"]
    if outcome.status is ResolutionStatus.RESOLVED and outcome.winner is not None:
        lines.append(f"Feature group: {outcome.winner.identity.qualname}")
        lines.append("Next step: request the feature; this environment resolves it unambiguously.")
    elif outcome.status is ResolutionStatus.AMBIGUOUS:
        survivors = ", ".join(
            candidate.identity.qualname
            for candidate in outcome.candidates
            if candidate.status is CandidateStatus.SURVIVOR
        )
        lines.append(f"Surviving candidates: {survivors}")
        lines.append("Next step: narrow the request with a feature_group scope, a domain, or a framework pin.")
    elif outcome.status is ResolutionStatus.FAILED:
        failing = ", ".join(sorted({failure.plugin.qualname for failure in outcome.failures}))
        lines.append(f"Failing plugin: {failing}")
        lines.append("Next step: fix or disable the failing plugin; a provider hook raised during resolution.")
    else:
        lines.append("Next step: check the feature name and load the providing plugin (PluginLoader.all()).")
    return "\n".join(lines)


def render_provider_view(outcome: ResolutionOutcome) -> str:
    """Per-candidate provider detail: identity, verdict, and every per-framework status."""
    lines = [f"Status: {outcome.status.value}"]
    for candidate in outcome.candidates:
        lines.append(f"{candidate.identity.render()}: {candidate.status.value}")
        for evaluation in candidate.frameworks:
            lines.append(f"  {evaluation.identity.render()}: {evaluation.status.value}")
        for rejection in candidate.rejections:
            lines.append(f"  rejection: {rejection.reason.value}")
    return "\n".join(lines)


def render_steward_view(outcome: ResolutionOutcome) -> dict[str, Any]:
    """Audit projection: the outcome's redacted plain-data payload, fingerprint included."""
    return outcome.to_payload()
