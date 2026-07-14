"""Core-only user surface. Importing this module needs no optional backend.

Backend import policy (the one place it is stated; backend modules only name their library):
each compute framework is published from one module per backend (mloda.user.pandas,
mloda.user.polars, ...). Importing a backend module always works, whether or not its library
is installed; a framework whose library is missing reports itself unavailable through
is_available() and is excluded from discovery.
"""

# Version
from mloda.core.version import get_mloda_version

# API
from mloda.core.api.request import mlodaAPI

# Resolved execution plan
from mloda.core.api.plan_info import PlanStep
from mloda.core.api.run_result import ResultStream, RunResult

# Features
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.domain import Domain

# Link & Index
from mloda.core.abstract_plugins.components.link import Link, JoinType, JoinSpec
from mloda.core.abstract_plugins.components.index.index import Index

# Filtering
from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.filter.single_filter import SingleFilter
from mloda.core.filter.filter_type_enum import FilterType

# Data access
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.credential import Credential

# Types
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode

# Plugin discovery
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector

# Plugin registry
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import (
    PluginRegistryCollisionError,
    register_plugin,
)

# Config loading
from mloda.core.api.feature_config.loader import load_features_from_config

# Plugin documentation / discovery
from mloda.core.api.plugin_docs import (
    get_feature_group_docs,
    get_compute_framework_docs,
    get_extender_docs,
    list_registered,
    resolve_feature,
)

__version__ = get_mloda_version()

mloda = mlodaAPI
stream_all = mlodaAPI.stream_all

__all__ = [
    # Version
    "__version__",
    # API
    "mlodaAPI",
    "mloda",
    # Resolved execution plan
    "PlanStep",
    "RunResult",
    "ResultStream",
    # Features
    "Feature",
    "Features",
    "FeatureName",
    "Options",
    "Domain",
    # Link & Index
    "Link",
    "JoinType",
    "JoinSpec",
    "Index",
    # Filtering
    "GlobalFilter",
    "SingleFilter",
    "FilterType",
    # Data access
    "DataAccessCollection",
    "Credential",
    # Types
    "DataType",
    "ParallelizationMode",
    # Plugin discovery
    "PluginLoader",
    "PluginCollector",
    # Plugin registry
    "PluginRegistryCollisionError",
    "register_plugin",
    # Streaming API
    "stream_all",
    # Config loading
    "load_features_from_config",
    # Plugin documentation / discovery
    "get_feature_group_docs",
    "get_compute_framework_docs",
    "get_extender_docs",
    "list_registered",
    "resolve_feature",
]
