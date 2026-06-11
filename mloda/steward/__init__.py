# Plugin inspection/metadata
from mloda.core.api.plugin_info import FeatureGroupInfo, ComputeFrameworkInfo, ExtenderInfo, ResolvedFeature

# Documentation/discovery
from mloda.core.api.plugin_docs import (
    get_feature_group_docs,
    get_compute_framework_docs,
    get_extender_docs,
    list_registered,
    resolve_feature,
)

# Function extenders (audit trails, monitoring, observability)
from mloda.core.abstract_plugins.function_extender import (
    Extender,
    ExtenderHook,
)

# Plugin registry administration
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry

# Plugin governance
from mloda.core.abstract_plugins.plugin_registry.plugin_policy import (
    ApprovalStatus,
    PluginPolicy,
    PluginPolicyViolationError,
)

__all__ = [
    # Plugin inspection
    "FeatureGroupInfo",
    "ComputeFrameworkInfo",
    "ExtenderInfo",
    "ResolvedFeature",
    # Documentation
    "get_feature_group_docs",
    "get_compute_framework_docs",
    "get_extender_docs",
    "list_registered",
    "resolve_feature",
    # Function extenders
    "Extender",
    "ExtenderHook",
    # Plugin registry administration
    "PluginRegistry",
    # Plugin governance
    "ApprovalStatus",
    "PluginPolicy",
    "PluginPolicyViolationError",
]
