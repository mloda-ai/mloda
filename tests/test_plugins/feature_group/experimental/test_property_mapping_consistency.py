"""Cross-plugin PROPERTY_MAPPING consistency sweep.

PROPERTY_MAPPING values are ``PropertySpec`` instances (issue #694): the dataclass
constructor enforces the per-spec invariants (known fields, flag types, strict needs a
value space), so this sweep pins what the type system cannot: every shipped plugin spec
IS a ``PropertySpec``, documents itself with a non-empty explanation, and enumerated
value spaces opt into strict validation.
"""

from typing import Any

import pytest

from mloda.provider import PropertySpec
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup

ALL_PLUGINS: list[type[Any]] = [
    AggregatedFeatureGroup,
    ClusteringFeatureGroup,
    MissingValueFeatureGroup,
    DimensionalityReductionFeatureGroup,
    ForecastingFeatureGroup,
    GeoDistanceFeatureGroup,
    NodeCentralityFeatureGroup,
    EncodingFeatureGroup,
    SklearnPipelineFeatureGroup,
    ScalingFeatureGroup,
    TextCleaningFeatureGroup,
    TimeWindowFeatureGroup,
]


@pytest.mark.parametrize("plugin_cls", ALL_PLUGINS, ids=lambda c: c.__name__)
class TestPropertyMappingConsistency:
    def test_every_spec_is_a_property_spec(self, plugin_cls: type[Any]) -> None:
        """Raw dict specs are retired: every PROPERTY_MAPPING value is a PropertySpec."""
        mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: list[str] = []
        for prop_key, spec in mapping.items():
            if not isinstance(spec, PropertySpec):
                violations.append(str(prop_key))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING entries that are not PropertySpec instances: {violations}"
        )

    def test_every_spec_has_a_nonempty_explanation(self, plugin_cls: type[Any]) -> None:
        """Every spec documents itself: the explanation is a non-empty string."""
        mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: list[str] = []
        for prop_key, spec in mapping.items():
            if not isinstance(spec.explanation, str) or not spec.explanation.strip():
                violations.append(str(prop_key))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING entries without a non-empty explanation: {violations}"
        )

    def test_enum_specs_have_strict_validation_true(self, plugin_cls: type[Any]) -> None:
        """A spec that enumerates its value space should enforce it.

        Specs with an element_validator delegate validation to the callable, and value
        spaces of fewer than two entries are documentation rather than an enumeration.
        """
        mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: list[str] = []
        for prop_key, spec in mapping.items():
            if spec.element_validator is not None:
                continue
            if spec.allowed_values is None or len(spec.allowed_values) < 2:
                continue
            if spec.strict_validation is not True:
                violations.append(str(prop_key))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING entries with enumerated values "
            f"should have strict_validation=True: {violations}"
        )
