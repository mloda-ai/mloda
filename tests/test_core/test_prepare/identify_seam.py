"""Test seam mirroring how production raises a resolution failure over IdentifyFeatureGroupClass.

The raising IdentifyFeatureGroupClass(...) constructor was removed: the engine and resolve_feature call
evaluate() and render the failure themselves. These helpers reproduce that engine seam so unit tests can
drive it without the deleted wrapper. Everything here is built on evaluate() + render_resolution_failure.
"""

from typing import Optional

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import (
    EvaluationResult,
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    render_resolution_failure,
)


def evaluate_or_raise(
    feature: Feature,
    accessible_plugins: FeatureGroupEnvironmentMapping,
    links: Optional[set[Link]] = None,
    data_access_collection: Optional[DataAccessCollection] = None,
) -> EvaluationResult:
    """Evaluate one feature and raise the typed error on failure, exactly as the engine seam does."""
    result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, links, data_access_collection)
    message = render_resolution_failure(result, feature)
    if message is not None:
        raise FeatureResolutionError(message, str(feature.name), result)
    return result


def identify_winner(
    feature: Feature,
    accessible_plugins: FeatureGroupEnvironmentMapping,
    links: Optional[set[Link]] = None,
    data_access_collection: Optional[DataAccessCollection] = None,
) -> tuple[type[FeatureGroup], set[type[ComputeFramework]]]:
    """The winning (feature_group, frameworks) pair, mirroring the removed IdentifyFeatureGroupClass.get()."""
    result = evaluate_or_raise(feature, accessible_plugins, links, data_access_collection)
    return next(iter(result.identified.items()))
