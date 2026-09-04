from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    # Subtype discriminator option key of the family (SUBTYPES.key), None for shape B and undeclared families.
    subtype_key: str | None = None
    # Sorted subtype universe: declared subtype values plus parametric family names.
    subtypes: list[str] = field(default_factory=list)
    # Sorted parametric subtype family names (e.g. "ntile" covering "ntile_2").
    parametric_subtypes: list[str] = field(default_factory=list)
    # Sorted supported subtypes per framework from compute_framework_definition(); empty for abstract classes.
    subtype_support: dict[str, list[str]] = field(default_factory=dict)
    # Message when the capability declaration is invalid, None for a legitimately empty matrix.
    subtype_error: str | None = None


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
    feature_group: type["FeatureGroup"] | None
    candidates: list[type["FeatureGroup"]]
    error: str | None
    # Frameworks supporting the feature, evaluated under the options passed to resolve_feature (empty by default).
    supported_compute_frameworks: list[str] = field(default_factory=list)
    # Frameworks rejecting the feature, evaluated under the options passed to resolve_feature (empty by default).
    unsupported_compute_frameworks: list[str] = field(default_factory=list)
    # Resolved subtype of the feature under the passed options, None when none resolves.
    subtype: str | None = None
    # Family name only when the subtype is a parametric instance (e.g. "ntile" for "ntile_2"), else None.
    subtype_family: str | None = None
