# Version
from mloda.core.version import get_mloda_version

# Plugin inspection/metadata
from mloda.core.api.plugin_info import FeatureGroupInfo, ComputeFrameworkInfo, ExtenderInfo, ResolvedFeature

# Resolved execution plan
from mloda.core.api.plan_info import PlanStep

# Plan lock file
from mloda.core.api.plan_lock import PlanLockMismatchError, check_plan_lock, write_plan_lock

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
    CompositeExtender,
    GateBypassError,
    LifecycleOutcome,
)
from mloda.core.abstract_plugins.plan_context import PlanContext
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.abstract_plugins.hook_context import HookContext, OutputSchema
from mloda.core.abstract_plugins.close_context import CloseContext
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig

# Server-verified tenant/project/principal context seam
from mloda.core.abstract_plugins.verified_context import verified_context

# Plugin registry administration
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry

# Optional-dependency import guards
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import traceback_blames_root

# Pickle safety for Extender authors (trial-pickle-and-warn-once for an injected sink)
from mloda.core.abstract_plugins.pickle_safety import (
    pickle_failure_reason,
    is_picklable,
    WarnOncePerInstance,
)

# Credential scrubbing for text an Extender logs
from mloda.core.abstract_plugins.components.credential_scrub import scrub_credentials

# Plugin governance
from mloda.core.abstract_plugins.plugin_registry.plugin_policy import (
    ApprovalStatus,
    PluginPolicy,
    PluginPolicyViolationError,
)

# Feature resolution
from mloda.core.prepare.identify_feature_group import FeatureResolutionError
from mloda.core.prepare.resolution_types import (
    ResolutionDiagnosis,
    ResolutionRecord,
)

__version__ = get_mloda_version()

__all__ = [
    # Version
    "__version__",
    # Plugin inspection
    "FeatureGroupInfo",
    "ComputeFrameworkInfo",
    "ExtenderInfo",
    "ResolvedFeature",
    # Resolved execution plan
    "PlanStep",
    # Plan lock file
    "write_plan_lock",
    "check_plan_lock",
    "PlanLockMismatchError",
    # Documentation
    "get_feature_group_docs",
    "get_compute_framework_docs",
    "get_extender_docs",
    "list_registered",
    "resolve_feature",
    # Function extenders
    "Extender",
    "ExtenderHook",
    "HookContext",
    "OutputSchema",
    "CompositeExtender",
    "GateBypassError",
    "LifecycleOutcome",
    "PlanContext",
    "RunContext",
    "CloseContext",
    "AsOfJoinConfig",
    # Server-verified tenant/project/principal context seam
    "verified_context",
    # Plugin registry administration
    "PluginRegistry",
    # Optional-dependency import guards
    "traceback_blames_root",
    # Pickle safety for Extender authors
    "pickle_failure_reason",
    "is_picklable",
    "WarnOncePerInstance",
    # Credential scrubbing
    "scrub_credentials",
    # Plugin governance
    "ApprovalStatus",
    "PluginPolicy",
    "PluginPolicyViolationError",
    # Feature resolution
    "FeatureResolutionError",
    "ResolutionRecord",
    "ResolutionDiagnosis",
]
