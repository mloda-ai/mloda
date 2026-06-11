"""Tests for the deployment-scoped governance policy value object (PluginPolicy).

Contract: mloda.core.abstract_plugins.plugin_registry.plugin_policy defines
ApprovalStatus (a str enum: unreviewed/approved/rejected), the
PluginPolicyViolationError exception, and the frozen dataclass PluginPolicy.
PluginPolicy.allows(key, module, approval) evaluates deny-first:
1) key in denied_keys, 2) module under denied_module_prefixes, 3) key in
allowed_keys bypasses the module-prefix constraint but NOT require_approval,
4) allowed_module_prefixes (None = unrestricted, () = nothing passes),
5) require_approval admits only ApprovalStatus.APPROVED.
"""

import dataclasses

import pytest

from mloda.core.abstract_plugins.plugin_registry.plugin_policy import (
    ApprovalStatus,
    PluginPolicy,
    PluginPolicyViolationError,
)


class TestPolicyModuleContract:
    def test_approval_status_is_str_enum_with_expected_values(self) -> None:
        assert issubclass(ApprovalStatus, str)
        assert ApprovalStatus.UNREVIEWED.value == "unreviewed"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.APPROVED == "approved"

    def test_violation_error_is_exception_subclass(self) -> None:
        assert issubclass(PluginPolicyViolationError, Exception)

    def test_policy_defaults_and_frozenness(self) -> None:
        policy = PluginPolicy()
        assert policy.allowed_module_prefixes is None
        assert policy.denied_module_prefixes == ()
        assert policy.allowed_keys == ()
        assert policy.denied_keys == ()
        assert policy.require_approval is False
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(policy, "require_approval", True)


class TestPolicyAllows:
    def test_default_policy_allows_any_key_module_and_approval(self) -> None:
        policy = PluginPolicy()
        assert policy.allows("any_pkg.mod:Cls", "any_pkg.mod", ApprovalStatus.UNREVIEWED) is True
        assert policy.allows("any_pkg.mod:Cls", "any_pkg.mod", ApprovalStatus.REJECTED) is True

    def test_denied_keys_deny_and_win_over_allowed_keys(self) -> None:
        policy = PluginPolicy(denied_keys=("pkg.mod:Bad",), allowed_keys=("pkg.mod:Bad",))
        assert policy.allows("pkg.mod:Bad", "pkg.mod", ApprovalStatus.APPROVED) is False
        assert policy.allows("pkg.mod:Good", "pkg.mod", ApprovalStatus.APPROVED) is True

    def test_denied_module_prefixes_deny_and_win_over_allowed_keys(self) -> None:
        policy = PluginPolicy(denied_module_prefixes=("evil_pkg",), allowed_keys=("evil_pkg.mod:Cls",))
        assert policy.allows("evil_pkg.mod:Cls", "evil_pkg.mod", ApprovalStatus.APPROVED) is False
        assert policy.allows("good_pkg.mod:Cls", "good_pkg.mod", ApprovalStatus.APPROVED) is True

    def test_allowed_module_prefixes_gate_modules(self) -> None:
        policy = PluginPolicy(allowed_module_prefixes=("mloda_plugins", "trusted_pkg"))
        assert policy.allows("trusted_pkg.mod:Cls", "trusted_pkg.mod", ApprovalStatus.UNREVIEWED) is True
        assert policy.allows("mloda_plugins.x:Cls", "mloda_plugins.x", ApprovalStatus.UNREVIEWED) is True
        assert policy.allows("outside_pkg.mod:Cls", "outside_pkg.mod", ApprovalStatus.UNREVIEWED) is False
        empty = PluginPolicy(allowed_module_prefixes=())
        assert empty.allows("any_pkg.mod:Cls", "any_pkg.mod", ApprovalStatus.APPROVED) is False

    def test_allowed_keys_bypass_module_prefix_but_not_require_approval(self) -> None:
        bypass = PluginPolicy(allowed_module_prefixes=("trusted_pkg",), allowed_keys=("outside_pkg.mod:Special",))
        assert bypass.allows("outside_pkg.mod:Special", "outside_pkg.mod", ApprovalStatus.UNREVIEWED) is True
        assert bypass.allows("outside_pkg.mod:Other", "outside_pkg.mod", ApprovalStatus.UNREVIEWED) is False
        strict = PluginPolicy(
            allowed_module_prefixes=("trusted_pkg",),
            allowed_keys=("outside_pkg.mod:Special",),
            require_approval=True,
        )
        assert strict.allows("outside_pkg.mod:Special", "outside_pkg.mod", ApprovalStatus.UNREVIEWED) is False
        assert strict.allows("outside_pkg.mod:Special", "outside_pkg.mod", ApprovalStatus.APPROVED) is True

    def test_require_approval_only_admits_approved(self) -> None:
        policy = PluginPolicy(require_approval=True)
        assert policy.allows("pkg.mod:Cls", "pkg.mod", ApprovalStatus.APPROVED) is True
        assert policy.allows("pkg.mod:Cls", "pkg.mod", ApprovalStatus.UNREVIEWED) is False
        assert policy.allows("pkg.mod:Cls", "pkg.mod", ApprovalStatus.REJECTED) is False
