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
    compute_frameworks: List[str]
    supported_feature_names: Set[str]
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
    wraps: List[str]


@dataclass
class ResolvedFeature:
    feature_name: str
    feature_group: Optional[Type["FeatureGroup"]]
    candidates: List[Type["FeatureGroup"]]
    error: Optional[str]
