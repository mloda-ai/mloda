"""Tests for the isolated_plugin_registry pytest fixture (issue #526, work item 2).

The fixture snapshots PluginRegistry.default(), yields it, and restores the
snapshot on teardown so tests cannot leak registrations into each other. It
also restores governance state: a policy set during the test is reverted to
the pre-test policy on teardown, and the per-instance policy-denial warning
dedup state is reset so the same denied key can warn again afterwards.
"""

import gc
import logging
import types
from typing import Any

import pytest

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.fixtures import isolated_plugin_registry
from mloda.core.abstract_plugins.plugin_registry.plugin_policy import PluginPolicy
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, register_module_plugins


class _FixtureThrowawayFG(FeatureGroup):
    pass


class _FixtureSmokeFG(FeatureGroup):
    pass


def test_isolated_plugin_registry_is_a_pytest_fixture() -> None:
    # pytest >= 8.4 exposes _fixture_function_marker on the FixtureFunctionDefinition,
    # older pytest set _pytestfixturefunction on the function itself.
    assert hasattr(isolated_plugin_registry, "_fixture_function_marker") or hasattr(
        isolated_plugin_registry, "_pytestfixturefunction"
    )
    assert hasattr(isolated_plugin_registry, "__wrapped__")


def test_isolated_plugin_registry_yields_default_and_restores_on_teardown() -> None:
    default = PluginRegistry.default()
    snap = default.snapshot()
    try:
        wrapped = getattr(isolated_plugin_registry, "__wrapped__")
        gen = wrapped()
        reg = next(gen)
        assert reg is PluginRegistry.default()
        key = reg.register(_FixtureThrowawayFG, name="_registry_fixture_throwaway")
        assert reg.get(key) is _FixtureThrowawayFG
        next(gen, None)
        assert PluginRegistry.default().get(key) is None
        assert PluginRegistry.default().is_registered(_FixtureThrowawayFG) is False
    finally:
        default.restore(snap)


def test_isolated_plugin_registry_smoke_usage(isolated_plugin_registry: PluginRegistry) -> None:
    key = isolated_plugin_registry.register(_FixtureSmokeFG, name="_registry_fixture_smoke")
    assert isolated_plugin_registry.get(key) is _FixtureSmokeFG
    assert isolated_plugin_registry is PluginRegistry.default()


def _denial_warnings(caplog: pytest.LogCaptureFixture, key: str) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and key in record.getMessage()
    ]


def test_isolated_plugin_registry_restores_pre_test_policy_on_teardown() -> None:
    default = PluginRegistry.default()
    original_policy = default.policy
    snap = default.snapshot()
    pre_test_policy = PluginPolicy(denied_keys=("_registry_fixture_pre_test_denied",))
    try:
        default.set_policy(pre_test_policy)
        wrapped = getattr(isolated_plugin_registry, "__wrapped__")
        gen = wrapped()
        reg = next(gen)
        in_test_policy = PluginPolicy(require_approval=True)
        reg.set_policy(in_test_policy)
        assert reg.policy is in_test_policy
        next(gen, None)
        assert PluginRegistry.default().policy is pre_test_policy
    finally:
        default.set_policy(original_policy)
        default.restore(snap)


def test_isolated_plugin_registry_resets_policy_denial_warn_dedup_on_teardown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    default = PluginRegistry.default()
    original_policy = default.policy
    snap = default.snapshot()
    module = types.ModuleType("registry_fixture_dedup_fake_mod")

    class _FixtureDedupDeniedFG(FeatureGroup):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Any, data_access_collection: Any = None
        ) -> bool:
            # Inert for other tests' feature resolution in this worker.
            return False

    _FixtureDedupDeniedFG.__module__ = module.__name__
    setattr(module, "_FixtureDedupDeniedFG", _FixtureDedupDeniedFG)
    denied_key = f"{module.__name__}:{_FixtureDedupDeniedFG.__qualname__}"
    denying_policy = PluginPolicy(denied_keys=(denied_key,))
    wrapped = getattr(isolated_plugin_registry, "__wrapped__")
    try:
        gen = wrapped()
        reg = next(gen)
        reg.set_policy(denying_policy)
        with caplog.at_level(logging.WARNING):
            keys = register_module_plugins(module)
        assert denied_key not in keys
        assert len(_denial_warnings(caplog, denied_key)) == 1
        next(gen, None)

        gen_again = wrapped()
        reg_again = next(gen_again)
        reg_again.set_policy(denying_policy)
        with caplog.at_level(logging.WARNING):
            keys_again = register_module_plugins(module)
        assert denied_key not in keys_again
        assert len(_denial_warnings(caplog, denied_key)) == 2, (
            "fixture teardown must reset the warn dedup so the same denied key warns again"
        )
        next(gen_again, None)
    finally:
        default.set_policy(original_policy)
        default.restore(snap)
        delattr(module, "_FixtureDedupDeniedFG")
        del _FixtureDedupDeniedFG, module
        gc.collect()
