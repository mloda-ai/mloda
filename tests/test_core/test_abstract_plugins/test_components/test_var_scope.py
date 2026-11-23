"""Typing and runtime pins for the VarScope values returned by activate()."""

from mloda.core.abstract_plugins.close_context import CloseContext
from mloda.core.abstract_plugins.compute_framework import ComputeFramework


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
