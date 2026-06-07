"""Chained-propagation edge case for context-key validation (TDD red phase).

Background
---------
``Options.update_with_protected_keys`` copies a SOURCE feature's
``propagate_context_keys`` values into the RECIPIENT's ``context`` (via
``self.context.update(propagating)``) but does NOT add those keys to the
recipient's own ``propagate_context_keys``. As a result a legitimately
propagated context key can sit in a recipient's context while being absent from
both the recipient's ``propagate_context_keys`` and (for an opt-in config-based
group) its derived schema.

At the validation boundary that key would then be flagged as an "unknown context
key" -- a spurious false positive for a key the framework itself propagated.

Intended end-state
------------------
A propagated context key must be TOLERATED. The primary test below merges a child
into a parent via the real ``update_with_protected_keys`` API and asserts the
parent resolves WITHOUT a spurious unknown-context-key error.

NOTE on the spec deviation (reported to the orchestrator):
The original task also proposed a directly-constructed ``Options`` with an empty
``propagate_context_keys`` and an unexplained context key, asserted to "not raise".
That form is un-greenable: the natural fix lives in ``update_with_protected_keys``,
which a directly-constructed object never executes, and the only alternative
(making ``validate_context_keys`` tolerate an unexplained key) would disable the
feature. It was therefore dropped in favor of the merge-based reproduction, which
exercises the actual fix site and goes green once propagation is corrected.
"""

from __future__ import annotations

from typing import Any, Optional

from mloda.provider import ComputeFramework
from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureChainParserMixin
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


class MockComputeFramework(ComputeFramework):
    """Mock compute framework for testing (never instantiated on this path)."""

    pass


FIXED_NAME = "propagation_opt_in_feature"


class ConfigOptInFG(FeatureChainParserMixin, FeatureGroup):
    """Config-based opt-in group whose schema does NOT list ``external_partition``."""

    PROPERTY_MAPPING = {
        "partition_by": {
            "explanation": "partition columns",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
        "frame_size": {
            "explanation": "rolling frame size",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "source features",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        name = str(feature_name) if isinstance(feature_name, FeatureName) else feature_name
        return name == FIXED_NAME

    @classmethod
    def context_key_schema(cls) -> Optional[dict[str, Any]]:
        return cls.derive_context_key_schema()

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestPropagatedContextKeyValidation:
    """Propagated-context-key tolerance at the validation boundary."""

    def test_propagated_context_key_is_tolerated_after_merge(self) -> None:
        """A context key that arrived via propagation must NOT be flagged as unknown.

        This reproduces the real mechanism: after ``update_with_protected_keys`` the
        propagated key lives in the parent's context but is absent from the parent's
        ``propagate_context_keys`` (and from the schema). Resolving the parent must
        not raise. RED signal: it currently does (see report).
        """
        parent = Options(
            context={"in_features": "x", "partition_by": "c"},
            propagate_context_keys=frozenset(),
        )
        child = Options(
            context={"in_features": "y", "external_partition": "v"},
            propagate_context_keys=frozenset({"external_partition"}),
        )

        parent.update_with_protected_keys(child)

        # Mechanism: the propagated value lands in the parent's context. This is
        # stable across the eventual fix, so it is a safe durable assertion.
        assert "external_partition" in parent.context

        feature = Feature(FIXED_NAME, options=parent)
        accessible_plugins: FeatureGroupEnvironmentMapping = {ConfigOptInFG: {MockComputeFramework}}

        # Intended end-state: no spurious "unknown context key 'external_partition'".
        result = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
        )
        assert ConfigOptInFG in result.feature_group_compute_framework_mapping

    def test_context_key_listed_in_propagate_set_is_allowed(self) -> None:
        """Backward-compat guard (NOT the propagation bug): a context key present in
        ``propagate_context_keys`` is allowed because validation treats the propagate
        set as extra allowed keys. Greenable by adding ``derive_context_key_schema``
        alone; does not depend on the propagation fix.
        """
        options = Options(
            context={"in_features": "x", "partition_by": "c", "external_partition": "v"},
            propagate_context_keys=frozenset({"external_partition"}),
        )
        feature = Feature(FIXED_NAME, options=options)
        accessible_plugins: FeatureGroupEnvironmentMapping = {ConfigOptInFG: {MockComputeFramework}}

        result = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
        )
        assert ConfigOptInFG in result.feature_group_compute_framework_mapping
