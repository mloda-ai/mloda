from dataclasses import dataclass
from typing import List, Optional, Set, Type, TYPE_CHECKING

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
