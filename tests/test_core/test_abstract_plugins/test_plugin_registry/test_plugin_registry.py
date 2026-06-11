"""Tests for the explicit plugin registry core (issue #526, work item 1).

Defines the contract for PluginRegistry: instance and process-global default,
register/unregister/get, collision handling, replace, type filtering,
snapshot/restore, the one-key-per-class invariant, and the module-level
register_plugin() convenience.
"""

import threading
from collections.abc import Iterator

import pytest

import mloda.core.abstract_plugins.plugin_registry.plugin_registry as plugin_registry_module
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import (
    PluginRegistry,
    PluginRegistryCollisionError,
    register_plugin,
)


class _RegistryTestFGA(FeatureGroup):
    pass


class _RegistryTestFGB(FeatureGroup):
    pass


class _RegistryTestCF(ComputeFramework):
    pass


class _RegistryTestExtender(Extender):
    pass


class _RegistryTestNotAPlugin:
    pass


def _default_key(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


@pytest.fixture
def default_registry_guard() -> Iterator[PluginRegistry]:
    """Snapshot the process-global default registry and restore it on teardown."""
    reg = PluginRegistry.default()
    snap = reg.snapshot()
    yield reg
    reg.restore(snap)


class TestPluginRegistryInstances:
    def test_instantiable(self) -> None:
        reg = PluginRegistry()
        assert isinstance(reg, PluginRegistry)

    def test_instances_have_independent_state(self) -> None:
        reg1 = PluginRegistry()
        reg2 = PluginRegistry()
        key = reg1.register(_RegistryTestFGA)
        assert reg1.get(key) is _RegistryTestFGA
        assert reg2.get(key) is None

    def test_default_returns_same_instance(self) -> None:
        assert PluginRegistry.default() is PluginRegistry.default()

    def test_default_is_not_a_fresh_instance(self) -> None:
        assert PluginRegistry.default() is not PluginRegistry()


class TestPluginRegistryRegister:
    def test_register_returns_default_key(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_RegistryTestFGA)
        assert key == _default_key(_RegistryTestFGA)

    def test_register_with_name_overrides_key(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_RegistryTestFGA, name="custom")
        assert key == "custom"
        assert reg.get("custom") is _RegistryTestFGA

    def test_register_source_defaults_to_manual(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_RegistryTestFGA)
        assert reg.get_entry(key).source == "manual"

    def test_register_records_source(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_RegistryTestFGA, source="entry_point")
        assert reg.get_entry(key).source == "entry_point"

    def test_register_non_plugin_class_raises_value_error(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(ValueError):
            reg.register(_RegistryTestNotAPlugin)

    def test_register_same_class_twice_is_noop(self) -> None:
        reg = PluginRegistry()
        key1 = reg.register(_RegistryTestFGA)
        key2 = reg.register(_RegistryTestFGA)
        assert key1 == key2
        assert reg.list_registered(FeatureGroup) == [_RegistryTestFGA]


class TestPluginRegistryCollision:
    def test_collision_error_is_value_error_subclass(self) -> None:
        assert issubclass(PluginRegistryCollisionError, ValueError)

    def test_different_class_same_key_raises_collision_error(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegistryTestFGA, name="dup_key")
        with pytest.raises(PluginRegistryCollisionError) as exc_info:
            reg.register(_RegistryTestFGB, name="dup_key")
        assert "dup_key" in str(exc_info.value)

    def test_collision_keeps_original_entry(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegistryTestFGA, name="dup_key")
        with pytest.raises(PluginRegistryCollisionError):
            reg.register(_RegistryTestFGB, name="dup_key")
        assert reg.get("dup_key") is _RegistryTestFGA

    def test_collision_error_message_teaches_replace_flag(self) -> None:
        """The collision message must tell users how to opt into replacement."""
        reg = PluginRegistry()
        reg.register(_RegistryTestFGA, name="dup_key")
        with pytest.raises(PluginRegistryCollisionError) as exc_info:
            reg.register(_RegistryTestFGB, name="dup_key")
        assert "replace=True" in str(exc_info.value), "collision error must mention replace=True so users learn the fix"

    def test_replace_true_replaces_entry(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegistryTestFGA, name="replace_key")
        key = reg.register(_RegistryTestFGB, name="replace_key", replace=True)
        assert key == "replace_key"
        assert reg.get("replace_key") is _RegistryTestFGB


class TestOneKeyPerClassInvariant:
    """A class holds exactly one registry key; aliasing the same class under a second key must raise."""

    def test_register_same_class_under_second_key_raises(self) -> None:
        reg = PluginRegistry()
        default_key = reg.register(_RegistryTestFGA)
        with pytest.raises(PluginRegistryCollisionError) as exc_info:
            reg.register(_RegistryTestFGA, name="alias")
        assert default_key in str(exc_info.value), "the one-key-per-class collision message must name the existing key"

    def test_register_default_key_after_custom_key_raises(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegistryTestFGA, name="custom")
        with pytest.raises(PluginRegistryCollisionError) as exc_info:
            reg.register(_RegistryTestFGA)
        assert "custom" in str(exc_info.value), "the one-key-per-class collision message must name the existing key"

    def test_second_key_collision_keeps_only_original_entry(self) -> None:
        reg = PluginRegistry()
        default_key = reg.register(_RegistryTestFGA)
        with pytest.raises(PluginRegistryCollisionError):
            reg.register(_RegistryTestFGA, name="alias")
        assert reg.get(default_key) is _RegistryTestFGA
        assert reg.get("alias") is None

    def test_replace_true_does_not_bypass_different_key_collision(self) -> None:
        """replace=True only resolves same-key-different-class; a second key for the same class still raises."""
        reg = PluginRegistry()
        default_key = reg.register(_RegistryTestFGA)
        with pytest.raises(PluginRegistryCollisionError) as exc_info:
            reg.register(_RegistryTestFGA, name="alias", replace=True)
        assert default_key in str(exc_info.value)
        assert reg.get("alias") is None

    def test_rename_via_unregister_then_register_succeeds(self) -> None:
        """The legitimate rename path: unregister the existing key, then register under the new name."""
        reg = PluginRegistry()
        reg.register(_RegistryTestFGA, name="old_name")
        reg.unregister("old_name")
        key = reg.register(_RegistryTestFGA, name="new_name")
        assert key == "new_name"
        assert reg.get("new_name") is _RegistryTestFGA
        assert reg.get("old_name") is None


class TestPluginRegistryLookup:
    def test_get_unknown_key_returns_none(self) -> None:
        reg = PluginRegistry()
        assert reg.get("does_not_exist") is None

    def test_unregister_removes_entry(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_RegistryTestFGA)
        reg.unregister(key)
        assert reg.get(key) is None

    def test_unregister_unknown_key_raises_value_error(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(ValueError):
            reg.unregister("does_not_exist")

    def test_get_entry_unknown_key_raises_value_error(self) -> None:
        """get_entry on an unknown key must raise ValueError, not a bare KeyError."""
        reg = PluginRegistry()
        with pytest.raises(ValueError):
            reg.get_entry("nope:Missing")

    def test_is_registered_true_for_registered_class(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegistryTestFGA, name="any_key")
        assert reg.is_registered(_RegistryTestFGA) is True

    def test_is_registered_false_for_unregistered_class(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegistryTestFGA)
        assert reg.is_registered(_RegistryTestFGB) is False

    def test_is_registered_false_after_unregister(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_RegistryTestFGA)
        reg.unregister(key)
        assert reg.is_registered(_RegistryTestFGA) is False


class TestPluginRegistryEntry:
    def test_get_entry_attributes(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_RegistryTestFGA)
        entry = reg.get_entry(key)
        assert entry.cls is _RegistryTestFGA
        assert entry.name == key
        assert entry.plugin_type is FeatureGroup
        assert entry.source_module == _RegistryTestFGA.__module__
        assert entry.source == "manual"

    def test_get_entry_plugin_type_compute_framework(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_RegistryTestCF)
        assert reg.get_entry(key).plugin_type is ComputeFramework

    def test_get_entry_plugin_type_extender(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_RegistryTestExtender)
        assert reg.get_entry(key).plugin_type is Extender


class TestPluginRegistryListRegistered:
    def test_list_registered_filters_by_plugin_type(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegistryTestFGA)
        reg.register(_RegistryTestCF)
        reg.register(_RegistryTestExtender)
        assert reg.list_registered(FeatureGroup) == [_RegistryTestFGA]
        assert reg.list_registered(ComputeFramework) == [_RegistryTestCF]
        assert reg.list_registered(Extender) == [_RegistryTestExtender]

    def test_list_registered_returns_list_sorted_by_key(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegistryTestFGB, name="z_key")
        reg.register(_RegistryTestFGA, name="a_key")
        result = reg.list_registered(FeatureGroup)
        assert isinstance(result, list)
        assert result == [_RegistryTestFGA, _RegistryTestFGB]

    def test_list_registered_lists_class_once_after_rejected_alias_attempt(self) -> None:
        """Aliasing now raises, so the class stays listed exactly once under its original key."""
        reg = PluginRegistry()
        reg.register(_RegistryTestFGA)
        with pytest.raises(PluginRegistryCollisionError):
            reg.register(_RegistryTestFGA, name="alias")
        result = reg.list_registered(FeatureGroup)
        assert result == [_RegistryTestFGA]

    def test_list_registered_is_registration_order_independent(self) -> None:
        reg_ab = PluginRegistry()
        reg_ab.register(_RegistryTestFGA)
        reg_ab.register(_RegistryTestFGB)
        reg_ba = PluginRegistry()
        reg_ba.register(_RegistryTestFGB)
        reg_ba.register(_RegistryTestFGA)
        assert reg_ab.list_registered(FeatureGroup) == reg_ba.list_registered(FeatureGroup)

    def test_list_registered_rejects_concrete_subclass(self) -> None:
        """A concrete subclass is not a valid plugin base type; silence would hide the bug."""
        reg = PluginRegistry()
        with pytest.raises(ValueError) as exc_info:
            reg.list_registered(_RegistryTestFGA)
        message = str(exc_info.value)
        assert "FeatureGroup" in message
        assert "ComputeFramework" in message
        assert "Extender" in message

    def test_list_registered_rejects_non_plugin_type(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(ValueError) as exc_info:
            reg.list_registered(int)
        message = str(exc_info.value)
        assert "FeatureGroup" in message
        assert "ComputeFramework" in message
        assert "Extender" in message

    def test_list_registered_accepts_exact_base_types(self) -> None:
        """The three exact plugin base types must keep working, even on an empty registry."""
        reg = PluginRegistry()
        assert reg.list_registered(FeatureGroup) == []
        assert reg.list_registered(ComputeFramework) == []
        assert reg.list_registered(Extender) == []

    def test_module_level_list_registered_wrapper_rejects_non_plugin_type(self) -> None:
        from mloda.core.api.plugin_docs import list_registered

        with pytest.raises(ValueError):
            list_registered(int)


class TestPluginRegistryDefaultThreadSafety:
    def test_default_lazy_init_is_thread_safe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """default() lazy init must be guarded by a module-level lock and return one instance.

        The lock-existence assertion makes this test fail deterministically before
        the implementation lands; the threaded part pins the observable contract.
        """
        lock = getattr(plugin_registry_module, "_default_lock", None)
        assert lock is not None, "plugin_registry must expose a module-level _default_lock for default()"
        assert isinstance(lock, type(threading.Lock())), "_default_lock must be a threading lock"

        monkeypatch.setattr(plugin_registry_module, "_default", None)
        thread_count = 16
        barrier = threading.Barrier(thread_count)
        results: list[PluginRegistry] = []

        def _call_default() -> None:
            barrier.wait()
            results.append(PluginRegistry.default())

        threads = [threading.Thread(target=_call_default) for _ in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == thread_count
        first = results[0]
        assert all(result is first for result in results), (
            "all threads racing default() must observe the same registry instance"
        )


class TestPluginRegistrySnapshotRestore:
    def test_restore_removes_entries_added_after_snapshot(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegistryTestFGA)
        snap = reg.snapshot()
        key_b = reg.register(_RegistryTestFGB)
        reg.restore(snap)
        assert reg.get(key_b) is None
        assert reg.is_registered(_RegistryTestFGB) is False

    def test_restore_brings_back_entries_removed_after_snapshot(self) -> None:
        reg = PluginRegistry()
        key_a = reg.register(_RegistryTestFGA)
        snap = reg.snapshot()
        reg.unregister(key_a)
        reg.restore(snap)
        assert reg.get(key_a) is _RegistryTestFGA

    def test_clear_empties_registry(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_RegistryTestFGA)
        reg.register(_RegistryTestCF)
        reg.clear()
        assert reg.get(key) is None
        assert reg.list_registered(FeatureGroup) == []
        assert reg.list_registered(ComputeFramework) == []


class TestModuleLevelRegisterPlugin:
    def test_register_plugin_writes_into_default_registry(self, default_registry_guard: PluginRegistry) -> None:
        key = register_plugin(_RegistryTestFGA)
        assert key == _default_key(_RegistryTestFGA)
        assert default_registry_guard.get(key) is _RegistryTestFGA
        assert PluginRegistry.default().get(key) is _RegistryTestFGA

    def test_register_plugin_accepts_same_keyword_arguments(self, default_registry_guard: PluginRegistry) -> None:
        key = register_plugin(_RegistryTestFGB, name="module_level_custom", source="loader")
        assert key == "module_level_custom"
        entry = default_registry_guard.get_entry("module_level_custom")
        assert entry.cls is _RegistryTestFGB
        assert entry.source == "loader"

    def test_old_register_name_removed_from_core_module(self) -> None:
        """The convenience function is renamed to register_plugin; the old name must be gone."""
        assert not hasattr(plugin_registry_module, "register"), (
            "mloda.core.abstract_plugins.plugin_registry.plugin_registry must not expose 'register'; "
            "the module-level convenience function is named register_plugin"
        )
