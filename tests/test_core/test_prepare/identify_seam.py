"""Canonical resolution test seam (os-014): target this, not IdentifyFeatureGroupClass internals.

- IdentifyFeatureGroupClass.evaluate(...) -> EvaluationResult: the authoritative matcher, non-raising for
  match outcomes (it still validates compute-framework pin cardinality up front).
- render_resolution_failure(result, feature) -> str | None: the pure failure renderer, None on success.
- evaluate_or_raise / identify_winner below: raise-on-failure, via the typed FeatureResolutionError.
- EvaluationResult fields (identified, criteria_matched, abstract_matched, candidate_frameworks,
  eliminations, facts) and the derived failure_kind property, for structured assertions.

evaluate_or_raise delegates to identify_feature_group.resolve_or_raise; target this seam, not the helper directly.

Exact failure-message wording is out of scope: it is inherently wording-coupled, so prefer asserting on
structured facts and reserve exact-string checks for tests whose contract is the wording itself.
"""

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import resolve_or_raise
from mloda.core.prepare.resolution_types import EvaluationResult


def evaluate_or_raise(
    feature: Feature,
    accessible_plugins: FeatureGroupEnvironmentMapping,
    links: set[Link] | None = None,
    data_access_collection: DataAccessCollection | None = None,
) -> EvaluationResult:
    """Evaluate one feature and raise the typed error on failure, exactly as the engine seam does."""
    return resolve_or_raise(feature, accessible_plugins, links, data_access_collection)


def identify_winner(
    feature: Feature,
    accessible_plugins: FeatureGroupEnvironmentMapping,
    links: set[Link] | None = None,
    data_access_collection: DataAccessCollection | None = None,
) -> tuple[type[FeatureGroup], set[type[ComputeFramework]]]:
    """The winning (feature_group, frameworks) pair, mirroring the removed IdentifyFeatureGroupClass.get()."""
    result = evaluate_or_raise(feature, accessible_plugins, links, data_access_collection)
    return next(iter(result.identified.items()))
