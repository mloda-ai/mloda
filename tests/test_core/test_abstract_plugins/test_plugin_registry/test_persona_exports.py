"""Persona-export matrix for the plugin registry public API (pre-merge tightening).

Contract: the module-level registration function is named register_plugin.
mloda.user exports register_plugin, list_registered, and PluginRegistryCollisionError,
but NOT PluginRegistry. mloda.provider exports register_plugin and
PluginRegistryCollisionError. mloda.steward exports PluginRegistry,
list_registered, and the governance objects PluginPolicy, ApprovalStatus, and
PluginPolicyViolationError. Every export is the identical object from its core
module and is listed in the namespace's __all__. Governance objects are
steward-only: mloda.user and mloda.provider gain nothing. PlanStep, the resolved
execution-plan record, is shared by mloda.user and mloda.steward.
"""

import importlib
from types import ModuleType

import pytest

import mloda.provider
import mloda.steward
import mloda.user

_NAMESPACES: dict[str, ModuleType] = {
    "user": mloda.user,
    "provider": mloda.provider,
    "steward": mloda.steward,
}

# Source modules are resolved lazily so a missing source module fails only its own matrix rows.
_SOURCE_MODULES: dict[str, str] = {
    "core_registry": "mloda.core.abstract_plugins.plugin_registry.plugin_registry",
    "plugin_docs": "mloda.core.api.plugin_docs",
    "core_policy": "mloda.core.abstract_plugins.plugin_registry.plugin_policy",
    "plan_info": "mloda.core.api.plan_info",
    "verified_context": "mloda.core.abstract_plugins.verified_context",
}


def _source_module(source_name: str) -> ModuleType:
    return importlib.import_module(_SOURCE_MODULES[source_name])


# (namespace, symbol, source module holding the canonical object)
EXPORT_MATRIX: list[tuple[str, str, str]] = [
    ("user", "register_plugin", "core_registry"),
    ("user", "list_registered", "plugin_docs"),
    ("user", "PluginRegistryCollisionError", "core_registry"),
    ("provider", "register_plugin", "core_registry"),
    ("provider", "PluginRegistryCollisionError", "core_registry"),
    ("steward", "PluginRegistry", "core_registry"),
    ("steward", "list_registered", "plugin_docs"),
    ("steward", "PluginPolicy", "core_policy"),
    ("steward", "ApprovalStatus", "core_policy"),
    ("steward", "PluginPolicyViolationError", "core_policy"),
    # Resolved execution plan (issue #647): the same PlanStep record for users and stewards.
    ("user", "PlanStep", "plan_info"),
    ("steward", "PlanStep", "plan_info"),
    # Verified context seam: steward-only, platform-integrator surface.
    ("steward", "verified_context", "verified_context"),
]

_MATRIX_IDS = [f"{namespace}-{symbol}" for namespace, symbol, _source in EXPORT_MATRIX]

_GOVERNANCE_SYMBOLS = ["PluginPolicy", "ApprovalStatus", "PluginPolicyViolationError"]


class TestPersonaExportMatrix:
    @pytest.mark.parametrize(("namespace_name", "symbol", "source_name"), EXPORT_MATRIX, ids=_MATRIX_IDS)
    def test_symbol_listed_in_namespace_all(self, namespace_name: str, symbol: str, source_name: str) -> None:
        namespace = _NAMESPACES[namespace_name]
        assert symbol in namespace.__all__, f"mloda.{namespace_name} must list '{symbol}' in __all__"

    @pytest.mark.parametrize(("namespace_name", "symbol", "source_name"), EXPORT_MATRIX, ids=_MATRIX_IDS)
    def test_symbol_is_identical_to_core_object(self, namespace_name: str, symbol: str, source_name: str) -> None:
        namespace = _NAMESPACES[namespace_name]
        source = _source_module(source_name)
        assert hasattr(source, symbol), f"{source.__name__} must define '{symbol}'"
        assert hasattr(namespace, symbol), f"mloda.{namespace_name} must expose '{symbol}'"
        assert getattr(namespace, symbol) is getattr(source, symbol), (
            f"mloda.{namespace_name}.{symbol} must be the identical object from {source.__name__}"
        )


class TestUserDoesNotExportPluginRegistry:
    def test_plugin_registry_absent_from_user_all(self) -> None:
        assert "PluginRegistry" not in mloda.user.__all__, (
            "mloda.user must not list PluginRegistry in __all__; it moves to mloda.steward"
        )

    def test_plugin_registry_not_a_user_attribute(self) -> None:
        assert not hasattr(mloda.user, "PluginRegistry"), (
            "mloda.user must not expose a PluginRegistry attribute; stewards use mloda.steward.PluginRegistry"
        )


class TestCurrentVerifiedContextStaysInternal:
    def test_current_verified_context_absent_from_steward_all(self) -> None:
        assert "current_verified_context" not in mloda.steward.__all__, (
            "mloda.steward must not list 'current_verified_context' in __all__; it is internal, "
            "used only by mlodaAPI._build_run_context"
        )

    def test_current_verified_context_not_a_steward_attribute(self) -> None:
        assert not hasattr(mloda.steward, "current_verified_context"), (
            "mloda.steward must not expose 'current_verified_context'"
        )


class TestVerifiedContextStaysInternal:
    """VerifiedContext itself is not needed by any public API consumer: only verified_context's
    kwargs are. It stays defined in verified_context.py but is not re-exported from mloda.steward."""

    def test_verified_context_absent_from_steward_all(self) -> None:
        assert "VerifiedContext" not in mloda.steward.__all__, (
            "mloda.steward must not list 'VerifiedContext' in __all__; only verified_context is exported"
        )

    def test_verified_context_not_a_steward_attribute(self) -> None:
        assert not hasattr(mloda.steward, "VerifiedContext"), "mloda.steward must not expose 'VerifiedContext'"


class TestGovernanceExportsAreStewardOnly:
    @pytest.mark.parametrize("symbol", _GOVERNANCE_SYMBOLS)
    def test_governance_symbol_absent_from_user_and_provider(self, symbol: str) -> None:
        for namespace_name in ("user", "provider"):
            namespace = _NAMESPACES[namespace_name]
            assert symbol not in namespace.__all__, f"mloda.{namespace_name} must not list '{symbol}' in __all__"
            assert not hasattr(namespace, symbol), f"mloda.{namespace_name} must not expose '{symbol}'"
