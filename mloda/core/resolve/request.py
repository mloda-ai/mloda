from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.resolve.identity import PluginIdentity


@dataclass(frozen=True)
class ResolutionRequestSnapshot:
    """Immutable capture of one feature request as per-feature matching consumes it.

    ``options`` is the SAME Options object as the feature: an opaque hook input the resolver never mutates.
    """

    feature_name: str
    domain: str | None
    feature_group_scope: str | type[FeatureGroup] | None
    framework_pin: str | None
    pinned_frameworks: tuple[type[ComputeFramework], ...]
    group_option_keys: frozenset[str]
    context_option_keys: frozenset[str]
    inherited_group_keys: frozenset[str]
    dependency_path: tuple[str, ...]
    links: tuple[Link, ...]
    data_access_collection: DataAccessCollection | None
    options: Options

    @classmethod
    def from_feature(
        cls,
        feature: Feature,
        links: set[Link] | frozenset[Link] | None = None,
        data_access_collection: DataAccessCollection | None = None,
        dependency_path: tuple[str, ...] = (),
    ) -> ResolutionRequestSnapshot:
        """Capture the matching-relevant request fields of ``feature`` at call time."""
        pinned: tuple[type[ComputeFramework], ...] = ()
        if feature.compute_frameworks:
            pinned = tuple(sorted(feature.compute_frameworks, key=PluginIdentity.from_class))
        return cls(
            feature_name=str(feature.name),
            domain=feature.domain.name if feature.domain is not None else None,
            feature_group_scope=feature.feature_group_scope,
            framework_pin=pinned[0].__name__ if len(pinned) == 1 else None,
            pinned_frameworks=pinned,
            group_option_keys=frozenset(feature.options.group),
            context_option_keys=frozenset(feature.options.context),
            inherited_group_keys=feature.options.inherited_group_keys,
            dependency_path=tuple(dependency_path),
            links=tuple(links) if links else (),
            data_access_collection=data_access_collection,
            options=feature.options,
        )

    def to_payload(self) -> dict[str, Any]:
        """Plain JSON data: option KEYS only, link count only; values, Options, and the collection are redacted."""
        scope = self.feature_group_scope
        scope_payload = scope if scope is None or isinstance(scope, str) else PluginIdentity.from_class(scope).render()
        return {
            "feature_name": self.feature_name,
            "domain": self.domain,
            "feature_group_scope": scope_payload,
            "framework_pin": self.framework_pin,
            "group_option_keys": sorted(self.group_option_keys),
            "context_option_keys": sorted(self.context_option_keys),
            "inherited_group_keys": sorted(self.inherited_group_keys),
            "dependency_path": list(self.dependency_path),
            "links": len(self.links),
        }
