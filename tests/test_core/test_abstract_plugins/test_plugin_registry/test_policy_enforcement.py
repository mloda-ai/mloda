"""Registration-time governance enforcement and approval metadata.

Contract: PluginRegistry.set_policy installs a deployment-scoped PluginPolicy,
exposed via the read-only `policy` property (default None = allow everything).
register() evaluates the policy after source normalization and before any state
mutation, against the would-be key, cls.__module__, and the approval being
registered. Denied manual registrations raise PluginPolicyViolationError naming
the key; denied loader/entry_point registrations register nothing, return None,
and warn once per key per registry instance. Entries carry owner/approval
metadata (approval strings normalized like source; invalid values raise
ValueError listing the valid ones). set_approval updates existing entries and
raises the get_entry error type for unknown names. The discovery funnels
(register_module_plugins, PluginLoader.load_entry_points) exclude denied
classes from their returned key lists without error.
"""

import gc
import importlib
import logging
import textwrap
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_policy import (
    ApprovalStatus,
    PluginPolicy,
    PluginPolicyViolationError,
)
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import (
    PluginRegistry,
    PluginSource,
    register_module_plugins,
    register_plugin,
)
from mloda.user import PluginLoader


class _PolicyEnfFGA(FeatureGroup):
    pass


class _PolicyEnfFGB(FeatureGroup):
    pass


class _PolicyEnfFGC(FeatureGroup):
    pass


class _PolicyEnfFGD(FeatureGroup):
    pass


class _DualCanonicalFG(FeatureGroup):
    pass


def _key(cls: type[Any]) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _warning_messages(caplog: pytest.LogCaptureFixture, key: str) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and key in record.getMessage()
    ]


@pytest.fixture
def default_registry_policy_guard() -> Iterator[PluginRegistry]:
    """Hand out the default registry and reset its policy on teardown.

    The autouse conftest fixture restores the entries snapshot; the policy is
    separate state and must be cleared explicitly.
    """
    registry = PluginRegistry.default()
    yield registry
    registry.set_policy(None)


class TestSetPolicy:
    def test_policy_property_defaults_to_none_and_roundtrips(self) -> None:
        reg = PluginRegistry()
        assert reg.policy is None
        policy = PluginPolicy(require_approval=True)
        reg.set_policy(policy)
        assert reg.policy is policy
        reg.set_policy(None)
        assert reg.policy is None
        with pytest.raises(AttributeError):
            setattr(reg, "policy", policy)

    def test_no_policy_allows_every_source(self) -> None:
        reg = PluginRegistry()
        assert reg.register(_PolicyEnfFGA) == _key(_PolicyEnfFGA)
        assert reg.register(_PolicyEnfFGB, source="loader") == _key(_PolicyEnfFGB)
        assert reg.register(_PolicyEnfFGC, source=PluginSource.ENTRY_POINT) == _key(_PolicyEnfFGC)


class TestPolicyDenialBySource:
    def test_manual_denied_raises_naming_key_and_mutates_nothing(self) -> None:
        reg = PluginRegistry()
        reg.set_policy(PluginPolicy(denied_keys=("governed_custom_key",)))
        with pytest.raises(PluginPolicyViolationError) as exc_info:
            reg.register(_PolicyEnfFGA, name="governed_custom_key")
        assert "governed_custom_key" in str(exc_info.value)
        assert reg.get("governed_custom_key") is None
        assert not reg.is_registered(_PolicyEnfFGA)
        assert reg.snapshot() == {}

    def test_invalid_source_string_raises_value_error_before_policy(self) -> None:
        """Enforcement runs after source normalization, so a bogus source wins over the policy."""
        reg = PluginRegistry()
        reg.set_policy(PluginPolicy(denied_keys=(_key(_PolicyEnfFGA),)))
        with pytest.raises(ValueError, match="bogus_source"):
            reg.register(_PolicyEnfFGA, source="bogus_source")

    def test_loader_denied_returns_none_and_warns_once_per_key(self, caplog: pytest.LogCaptureFixture) -> None:
        reg = PluginRegistry()
        denied_a = _key(_PolicyEnfFGA)
        denied_b = _key(_PolicyEnfFGB)
        reg.set_policy(PluginPolicy(denied_keys=(denied_a, denied_b)))
        with caplog.at_level(logging.WARNING):
            assert reg.register(_PolicyEnfFGA, source="loader") is None
            assert len(_warning_messages(caplog, denied_a)) == 1
            assert reg.register(_PolicyEnfFGA, source="loader") is None
            assert len(_warning_messages(caplog, denied_a)) == 1, "second denial of the same key must not warn again"
            assert reg.register(_PolicyEnfFGB, source="loader") is None
            assert len(_warning_messages(caplog, denied_b)) == 1, "a different denied key warns on its own"
        assert not reg.is_registered(_PolicyEnfFGA)
        assert not reg.is_registered(_PolicyEnfFGB)

    def test_entry_point_denied_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        reg = PluginRegistry()
        denied = _key(_PolicyEnfFGC)
        reg.set_policy(PluginPolicy(denied_keys=(denied,)))
        with caplog.at_level(logging.WARNING):
            assert reg.register(_PolicyEnfFGC, source=PluginSource.ENTRY_POINT) is None
        assert len(_warning_messages(caplog, denied)) == 1
        assert reg.get(denied) is None


class TestApprovalMetadata:
    def test_entry_owner_and_approval_defaults(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_PolicyEnfFGA)
        assert key is not None
        entry = reg.get_entry(key)
        assert entry.owner is None
        assert entry.approval is ApprovalStatus.UNREVIEWED

    def test_register_stores_owner_and_normalizes_approval(self) -> None:
        reg = PluginRegistry()
        key_a = reg.register(_PolicyEnfFGA, owner="alice", approval=ApprovalStatus.APPROVED)
        assert key_a is not None
        entry_a = reg.get_entry(key_a)
        assert entry_a.owner == "alice"
        assert entry_a.approval is ApprovalStatus.APPROVED
        key_b = reg.register(_PolicyEnfFGB, approval="rejected")
        assert key_b is not None
        assert reg.get_entry(key_b).approval is ApprovalStatus.REJECTED

    def test_register_invalid_approval_string_raises_listing_valid_values(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(ValueError) as exc_info:
            reg.register(_PolicyEnfFGA, approval="bogus_approval")
        message = str(exc_info.value)
        assert "bogus_approval" in message
        for valid in ("unreviewed", "approved", "rejected"):
            assert valid in message, f"the error must list the valid approval value '{valid}'"
        assert not reg.is_registered(_PolicyEnfFGA)

    def test_require_approval_policy_blocks_unapproved_and_admits_approved(self) -> None:
        reg = PluginRegistry()
        reg.set_policy(PluginPolicy(require_approval=True))
        with pytest.raises(PluginPolicyViolationError):
            reg.register(_PolicyEnfFGA)
        assert not reg.is_registered(_PolicyEnfFGA)
        key = reg.register(_PolicyEnfFGB, approval=ApprovalStatus.APPROVED)
        assert key == _key(_PolicyEnfFGB)
        assert reg.get_entry(_key(_PolicyEnfFGB)).approval is ApprovalStatus.APPROVED


class TestSetApproval:
    def test_set_approval_updates_approval_and_owner(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_PolicyEnfFGA, owner="alice")
        assert key is not None
        reg.set_approval(key, ApprovalStatus.APPROVED, owner="bob")
        entry = reg.get_entry(key)
        assert entry.approval is ApprovalStatus.APPROVED
        assert entry.owner == "bob"
        assert entry.cls is _PolicyEnfFGA
        assert entry.name == key

    def test_set_approval_normalizes_string_and_keeps_owner_when_omitted(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_PolicyEnfFGA, owner="alice")
        assert key is not None
        reg.set_approval(key, "rejected")
        entry = reg.get_entry(key)
        assert entry.approval is ApprovalStatus.REJECTED
        assert entry.owner == "alice"

    def test_set_approval_unknown_name_raises_value_error(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(ValueError):
            reg.set_approval("nope:Missing", ApprovalStatus.APPROVED)


class TestModuleLevelRegisterPluginForwarding:
    def test_register_plugin_forwards_owner_and_approval(self) -> None:
        key = register_plugin(_PolicyEnfFGD, owner="carol", approval="approved")
        assert key is not None
        entry = PluginRegistry.default().get_entry(key)
        assert entry.owner == "carol"
        assert entry.approval is ApprovalStatus.APPROVED


class TestRegisterModulePluginsSkipsDenied:
    def test_register_module_plugins_excludes_denied(self, default_registry_policy_guard: PluginRegistry) -> None:
        registry = default_registry_policy_guard
        module = types.ModuleType("policy_gate_fake_mod")

        class _ModAllowedFG(FeatureGroup):
            @classmethod
            def match_feature_group_criteria(
                cls, feature_name: Any, options: Any, data_access_collection: Any = None
            ) -> bool:
                # Inert for other tests' feature resolution in this worker.
                return False

        class _ModDeniedFG(FeatureGroup):
            @classmethod
            def match_feature_group_criteria(
                cls, feature_name: Any, options: Any, data_access_collection: Any = None
            ) -> bool:
                return False

        _ModAllowedFG.__module__ = module.__name__
        _ModDeniedFG.__module__ = module.__name__
        setattr(module, "_ModAllowedFG", _ModAllowedFG)
        setattr(module, "_ModDeniedFG", _ModDeniedFG)
        allowed_key = f"{module.__name__}:{_ModAllowedFG.__qualname__}"
        denied_key = f"{module.__name__}:{_ModDeniedFG.__qualname__}"
        try:
            registry.set_policy(PluginPolicy(denied_keys=(denied_key,)))
            keys = register_module_plugins(module)
            assert all(isinstance(key, str) for key in keys), "denied classes must not surface as None results"
            assert allowed_key in keys
            assert denied_key not in keys
            assert registry.is_registered(_ModAllowedFG)
            assert not registry.is_registered(_ModDeniedFG)
        finally:
            # Drop every strong ref to the fake-module doubles so they cannot poison
            # other tests in this worker via FeatureGroup.__subclasses__(); the autouse
            # conftest fixture restores the registry snapshot after the test.
            registry.clear()
            delattr(module, "_ModAllowedFG")
            delattr(module, "_ModDeniedFG")
            del _ModAllowedFG, _ModDeniedFG, module
            gc.collect()


_POLICY_EP_MANIFEST = """
    from mloda.core.abstract_plugins.feature_group import FeatureGroup


    class AllowedEpPolicyFeatureGroup(FeatureGroup):
        pass


    class DeniedEpPolicyFeatureGroup(FeatureGroup):
        pass


    FEATURE_GROUPS = [AllowedEpPolicyFeatureGroup, DeniedEpPolicyFeatureGroup]
"""


def _build_distribution(base_dir: Path, pkg_name: str, manifest_source: str, entry_points_txt: str) -> None:
    """Create a real on-disk distribution: importable package plus dist-info metadata."""
    pkg_dir = base_dir / pkg_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "manifest.py").write_text(textwrap.dedent(manifest_source))
    dist_info = base_dir / f"{pkg_name}-1.0.0.dist-info"
    dist_info.mkdir()
    metadata = f"Metadata-Version: 2.1\nName: {pkg_name.replace('_', '-')}\nVersion: 1.0.0\n"
    (dist_info / "METADATA").write_text(metadata)
    (dist_info / "entry_points.txt").write_text(textwrap.dedent(entry_points_txt))
    importlib.invalidate_caches()


class TestLoadEntryPointsSkipsDenied:
    def test_load_entry_points_excludes_denied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, default_registry_policy_guard: PluginRegistry
    ) -> None:
        pkg = "eptest_policy_denied_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _POLICY_EP_MANIFEST,
            f"""
            [mloda.feature_groups]
            demo = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        allowed_key = f"{pkg}.manifest:AllowedEpPolicyFeatureGroup"
        denied_key = f"{pkg}.manifest:DeniedEpPolicyFeatureGroup"
        registry = default_registry_policy_guard
        registry.set_policy(PluginPolicy(denied_keys=(denied_key,)))

        keys = PluginLoader().load_entry_points()

        assert allowed_key in keys
        assert denied_key not in keys
        assert all(isinstance(key, str) for key in keys), "denied classes must not surface as None results"
        assert registry.get(denied_key) is None
        assert registry.get_entry(allowed_key).source == PluginSource.ENTRY_POINT


class TestDualImportAwareness:
    def test_alias_module_path_gets_distinct_key_and_policy_allows_only_canonical(self) -> None:
        """Two class objects for "the same" plugin under two module paths are two keys;
        an allowed_module_prefixes policy covering only the canonical path splits them."""
        canonical_module = _DualCanonicalFG.__module__
        unrestricted = PluginRegistry()
        governed = PluginRegistry()

        class _DualAliasFG(FeatureGroup):
            @classmethod
            def match_feature_group_criteria(
                cls, feature_name: Any, options: Any, data_access_collection: Any = None
            ) -> bool:
                # Inert for other tests' feature resolution in this worker.
                return False

        _DualAliasFG.__module__ = "dual_import_alias_pkg.shadowed_manifest"
        try:
            canonical_key = unrestricted.register(_DualCanonicalFG)
            alias_key = unrestricted.register(_DualAliasFG)
            assert canonical_key is not None
            assert alias_key is not None
            assert canonical_key != alias_key, "dual-imported class objects must get two distinct keys"

            governed.set_policy(PluginPolicy(allowed_module_prefixes=(canonical_module,)))
            assert governed.register(_DualCanonicalFG) == canonical_key
            assert governed.register(_DualAliasFG, source="loader") is None
            assert governed.is_registered(_DualCanonicalFG)
            assert not governed.is_registered(_DualAliasFG)
        finally:
            unrestricted.clear()
            governed.clear()
            del _DualAliasFG
            gc.collect()
