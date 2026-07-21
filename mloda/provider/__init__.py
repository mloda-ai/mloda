# Data Provider: Base classes for building features
#
# Building a feature group plugin
# --------------------------------
# Feature groups that chain (derive from other features) should inherit from
# both FeatureChainParserMixin and FeatureGroup. The mixin handles feature
# name parsing, PROPERTY_MAPPING validation, and input_features resolution.
# See FeatureGroup's docstring and docs/in_depth/property-mapping.md.
#
#     from mloda.provider import FeatureChainParserMixin, FeatureGroup, DefaultOptionKeys
#
#     class MyPlugin(FeatureChainParserMixin, FeatureGroup):
#         PREFIX_PATTERN = r".*__([\w]+)_my_op$"
#         PROPERTY_MAPPING = { ... }
#         def calculate_feature(cls, data, features): ...
#
# Primary-source feature groups (no input features) subclass FeatureGroup
# directly and implement input_features and match_feature_group_criteria.
#
# This module is core-only. Concrete compute frameworks come from mloda.user.<backend>.
#
from mloda.core.abstract_plugins.feature_group import FeatureGroup as FeatureGroup

# Versioning
from mloda.core.abstract_plugins.components.base_feature_group_version import BaseFeatureGroupVersion
from mloda.core.version import get_mloda_version
from mloda.core.abstract_plugins.compute_framework import ComputeFramework as ComputeFramework
from mloda.core.abstract_plugins.compute_framework import EmptyResultError as EmptyResultError

# Utilities
from mloda.core.abstract_plugins.components.hashable_dict import HashableDict
from mloda.core.abstract_plugins.components.utils import get_all_subclasses

# Feature set (internal computation container)
from mloda.core.abstract_plugins.components.feature_set import FeatureSet

# Input data classes
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.core.abstract_plugins.components.input_data.input_data_descriptor import InputDataDescriptor
from mloda.core.abstract_plugins.components.input_data.api.api_input_data import ApiInputData
from mloda.core.abstract_plugins.components.input_data.api.api_input_data_feature import ApiInputDataFeature
from mloda.core.abstract_plugins.components.input_data.api.base_api_data import BaseApiData
from mloda.core.abstract_plugins.components.input_data.api.api_input_data_collection import ApiInputDataCollection
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator

# Match data
from mloda.core.abstract_plugins.components.match_data.match_data import MatchData

# Artifact
from mloda.core.abstract_plugins.components.base_artifact import BaseArtifact

# Validators
from mloda.core.abstract_plugins.components.base_validator import BaseValidator
from mloda.core.abstract_plugins.components.validators.feature_validator import FeatureValidator
from mloda.core.abstract_plugins.components.validators.feature_set_validator import FeatureSetValidator
from mloda.core.abstract_plugins.components.validators.options_validator import OptionsValidator
from mloda.core.abstract_plugins.components.validators.link_validator import LinkValidator
from mloda.core.abstract_plugins.components.validators.datatype_validator import (
    DataTypeValidator,
    DataTypeMismatchError,
)

# Option keys
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

# Feature chaining
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    CHAIN_SEPARATOR,
    COLUMN_SEPARATOR,
    INPUT_SEPARATOR,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.parsed_feature_name import ParsedFeatureName
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import (
    NO_DEFAULT,
    PropertySpec,
    is_no_default,
    is_positive_int,
    property_spec,
)

# Subtype declaration
from mloda.core.abstract_plugins.components.subtype_declaration import SubtypeDeclaration

# Transformers
from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer

# Plugin registry
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import (
    PluginRegistryCollisionError,
    register_plugin,
)

# Engines
from mloda.core.filter.filter_engine import BaseFilterEngine
from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine
from mloda.core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine

# Feature resolution debugging
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.prepare.identify_feature_group import (
    FeatureResolutionError,
    ResolutionDiagnosis,
    ResolutionRecord,
)

__version__ = get_mloda_version()

__all__ = [
    # Version
    "__version__",
    # Base classes
    "FeatureGroup",
    # Versioning
    "BaseFeatureGroupVersion",
    "ComputeFramework",
    "EmptyResultError",
    # Utilities
    "HashableDict",
    "get_all_subclasses",
    # Feature set
    "FeatureSet",
    # Input data
    "BaseInputData",
    "FileSource",
    "InputDataDescriptor",
    "ApiInputData",
    "ApiInputDataFeature",
    "BaseApiData",
    "ApiInputDataCollection",
    "DataCreator",
    # Match data
    "MatchData",
    # Artifact
    "BaseArtifact",
    # Validators
    "BaseValidator",
    "FeatureValidator",
    "FeatureSetValidator",
    "OptionsValidator",
    "LinkValidator",
    "DataTypeValidator",
    "DataTypeMismatchError",
    # Option keys
    "DefaultOptionKeys",
    # Feature chaining
    "FeatureChainParser",
    "CHAIN_SEPARATOR",
    "COLUMN_SEPARATOR",
    "INPUT_SEPARATOR",
    "FeatureChainParserMixin",
    "ParsedFeatureName",
    "PropertySpec",
    "is_no_default",
    "is_positive_int",
    "property_spec",
    "NO_DEFAULT",
    # Subtype declaration
    "SubtypeDeclaration",
    # Transformers
    "BaseTransformer",
    "ComputeFrameworkTransformer",
    # Plugin registry
    "PluginRegistryCollisionError",
    "register_plugin",
    # Engines
    "BaseFilterEngine",
    "BaseMaskEngine",
    "BaseMergeEngine",
    # Feature resolution debugging
    "resolve_feature",
    "FeatureResolutionError",
    "ResolutionRecord",
    "ResolutionDiagnosis",
]
