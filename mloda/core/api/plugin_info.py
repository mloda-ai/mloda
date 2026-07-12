from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.feature_group import FeatureGroup


@dataclass
class FeatureGroupInfo:
    name: str
    description: str
    version: str
    module: str
    compute_frameworks: list[str]
    supported_feature_names: set[str]
    prefix: str
    # PROPERTY_MAPPING key carrying the subtype discriminator, None without a subtype dimension.
    subtype_key: Optional[str] = None
    # Every subtype the family declares, sorted.
    subtypes: list[str] = field(default_factory=list)
    # Compute framework class name -> supported subtypes, sorted.
    subtype_support: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ComputeFrameworkInfo:
    name: str
    description: str
    module: str
    is_available: bool
    expected_data_framework: str
    has_merge_engine: bool
    has_filter_engine: bool


@dataclass
class ExtenderInfo:
    name: str
    description: str
    module: str
    wraps: list[str]


@dataclass
class ResolvedFeature:
    feature_name: str
    feature_group: Optional[type["FeatureGroup"]]
    candidates: list[type["FeatureGroup"]]
    error: Optional[str]
    # Frameworks supporting the feature, evaluated under default options.
    supported_compute_frameworks: list[str] = field(default_factory=list)
    # Frameworks rejecting the feature, evaluated under default options.
    unsupported_compute_frameworks: list[str] = field(default_factory=list)
    # Subtype resolved from the feature name/options by the resolved feature group.
    subtype: Optional[str] = None
