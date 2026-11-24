"""Typing and runtime pins for the VarScope values returned by activate()."""

from mloda.core.abstract_plugins.close_context import CloseContext
from mloda.core.abstract_plugins.components.input_data.claim_route import ChainCheck, chain_check_scope
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.implementation import accessible_implementations
from mloda.core.abstract_plugins.verified_context import verified_context


def test_close_context_activate_yields_non_optional() -> None:
    close_context = CloseContext(deadline=1.0, reason="stop")

    with close_context.activate() as active:
        assert active.reason == "stop"

    assert active.reason == "stop"
    assert active is close_context


def test_compute_framework_activate_yields_non_optional() -> None:
    framework = ComputeFramework()

    with framework.activate() as active:
        assert active.column_names == set()

    assert active.column_names == set()
    assert active is framework


def test_verified_context_yields_non_optional() -> None:
    with verified_context(tenant_id="t") as vc:
        assert vc.tenant_id == "t"


def test_chain_check_scope_yields_non_optional() -> None:
    with chain_check_scope(ChainCheck("name")) as check:
        assert check.probes == {}


def test_accessible_implementations_yields_non_optional() -> None:
    with accessible_implementations([]) as implementations:
        assert len(implementations) == 0
