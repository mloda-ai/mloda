from dataclasses import dataclass
from typing import List, Set


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
