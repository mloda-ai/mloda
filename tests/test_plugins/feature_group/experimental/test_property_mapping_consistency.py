from typing import Any, Dict, List, Tuple, Type

import pytest

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

ALL_PLUGINS: List[Type[Any]] = [
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

METADATA_KEYS: Tuple[Any, ...] = (
    DefaultOptionKeys.context,
    DefaultOptionKeys.default,
    DefaultOptionKeys.strict_validation,
    DefaultOptionKeys.validation_function,
    DefaultOptionKeys.required_when,
    DefaultOptionKeys.type_validator,
    DefaultOptionKeys.in_features,
    DefaultOptionKeys.group,
    "explanation",
)

DOK_VALUES: List[str] = [dok.value for dok in DefaultOptionKeys]


@pytest.mark.parametrize("plugin_cls", ALL_PLUGINS, ids=lambda c: c.__name__)
class TestPropertyMappingConsistency:
    def test_every_entry_has_explicit_strict_validation(self, plugin_cls: Type[Any]) -> None:
        mapping: Dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: List[str] = []
        for prop_key, entry in mapping.items():
            if not isinstance(entry, dict):
                continue
            if DefaultOptionKeys.strict_validation not in entry:
                violations.append(str(prop_key))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING entries missing DefaultOptionKeys.strict_validation: {violations}"
        )

    def test_no_raw_string_metadata_keys(self, plugin_cls: Type[Any]) -> None:
        mapping: Dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: List[Tuple[str, str]] = []
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

    def test_enum_entries_have_strict_validation_true(self, plugin_cls: Type[Any]) -> None:
        mapping: Dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: List[str] = []
        for prop_key, entry in mapping.items():
            if not isinstance(entry, dict):
                continue
            if DefaultOptionKeys.validation_function in entry:
                continue
            non_metadata_keys = [
                k
                for k in entry
                if k not in METADATA_KEYS and not (k in DOK_VALUES and not isinstance(k, DefaultOptionKeys))
            ]
            if len(non_metadata_keys) < 2:
                continue
            sv_value = entry.get(DefaultOptionKeys.strict_validation)
            if sv_value is not True:
                violations.append(str(prop_key))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING entries with enumerated values "
            f"should have strict_validation=True: {violations}"
        )
