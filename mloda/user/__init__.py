import importlib
from typing import TYPE_CHECKING, Any

# API
from mloda.core.api.request import mlodaAPI

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

mloda = mlodaAPI
stream_all = mlodaAPI.stream_all

# Lazy compute-framework exports (issue #649): resolved on demand so that
# ``import mloda.user`` stays dependency-free and optional backends propagate
# ModuleNotFoundError only at access time.
_LAZY_COMPUTE_FRAMEWORKS: dict[str, str] = {
    "PythonDictFramework": "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework",
    "PandasDataFrame": "mloda_plugins.compute_framework.base_implementations.pandas.dataframe",
    "PolarsDataFrame": "mloda_plugins.compute_framework.base_implementations.polars.dataframe",
    "PolarsLazyDataFrame": "mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe",
    "PyArrowTable": "mloda_plugins.compute_framework.base_implementations.pyarrow.table",
    "SqliteFramework": "mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework",
    "DuckDBFramework": "mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework",
    "SparkFramework": "mloda_plugins.compute_framework.base_implementations.spark.spark_framework",
    "IcebergFramework": "mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework",
}

if TYPE_CHECKING:
    # Static re-exports for mypy/tooling; not imported at runtime.
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import (
        DuckDBFramework as DuckDBFramework,
    )
    from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import (
        IcebergFramework as IcebergFramework,
    )
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame as PandasDataFrame
    from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame as PolarsDataFrame
    from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import (
        PolarsLazyDataFrame as PolarsLazyDataFrame,
    )
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable as PyArrowTable
    from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
        PythonDictFramework as PythonDictFramework,
    )
    from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import (
        SparkFramework as SparkFramework,
    )
    from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import (
        SqliteFramework as SqliteFramework,
    )


def __getattr__(name: str) -> Any:
    # PEP 562 lazy resolution of optional compute-framework classes.
    module_path = _LAZY_COMPUTE_FRAMEWORKS.get(name)
    if module_path is not None:
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # API
    "mlodaAPI",
    "mloda",
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
    # Compute frameworks (lazy, issue #649)
    "PythonDictFramework",
    "PandasDataFrame",
    "PolarsDataFrame",
    "PolarsLazyDataFrame",
    "PyArrowTable",
    "SqliteFramework",
    "DuckDBFramework",
    "SparkFramework",
    "IcebergFramework",
]
