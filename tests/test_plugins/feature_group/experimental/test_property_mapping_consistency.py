from typing import Any

import pytest

from mloda.core.abstract_plugins.components.default_options_key import PROPERTY_SPEC_KEYS
from mloda.provider import DefaultOptionKeys
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

DOK_VALUES: list[str] = [dok.value for dok in DefaultOptionKeys]


@pytest.mark.parametrize("plugin_cls", ALL_PLUGINS, ids=lambda c: c.__name__)
class TestPropertyMappingConsistency:
    def test_every_entry_has_explicit_strict_validation(self, plugin_cls: type[Any]) -> None:
        mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: list[str] = []
        for prop_key, entry in mapping.items():
            if not isinstance(entry, dict):
                continue
            if DefaultOptionKeys.strict_validation not in entry:
                violations.append(str(prop_key))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING entries missing DefaultOptionKeys.strict_validation: {violations}"
        )

    def test_no_raw_string_metadata_keys(self, plugin_cls: type[Any]) -> None:
        mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: list[tuple[str, str]] = []
        for prop_key, entry in mapping.items():
            if not isinstance(entry, dict):
                continue
            for key in entry:
                if key in DOK_VALUES and not isinstance(key, DefaultOptionKeys):
                    violations.append((str(prop_key), str(key)))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING uses raw strings where "
            f"DefaultOptionKeys enum members should be used: {violations}"
        )

    def test_only_known_spec_keys(self, plugin_cls: type[Any]) -> None:
        """Every key of every spec is part of the spec schema.

        Derived from the single source of truth, so a plugin cannot reintroduce a bare value
        entry (the retired flattened form) or a typo'd flag without this failing.
        """
        mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: list[tuple[str, str]] = []
        for prop_key, entry in mapping.items():
            if not isinstance(entry, dict):
                continue
            for key in entry:
                if key not in PROPERTY_SPEC_KEYS:
                    violations.append((str(prop_key), str(key)))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING carries keys outside the spec schema "
            f"(accepted values belong under allowed_values): {violations}"
        )

    def test_enum_entries_have_strict_validation_true(self, plugin_cls: type[Any]) -> None:
        mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: list[str] = []
        for prop_key, entry in mapping.items():
            if not isinstance(entry, dict):
                continue
            if DefaultOptionKeys.element_validator in entry:
                continue
            allowed_values = entry.get(DefaultOptionKeys.allowed_values) or {}
            if len(allowed_values) < 2:
                continue
            sv_value = entry.get(DefaultOptionKeys.strict_validation)
            if sv_value is not True:
                violations.append(str(prop_key))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING entries with enumerated values "
            f"should have strict_validation=True: {violations}"
        )
