"""Pins the public rename of ``_CompositeExtender`` -> ``CompositeExtender``.

Covers direct import off the underscored name, re-export from ``mloda.steward``,
``__all__`` membership, absence of the old private name, and a behavioral sanity
check that the renamed class still chains extenders correctly.
"""

from typing import Any
import mloda.core.abstract_plugins.function_extender as function_extender_module

from mloda.core.abstract_plugins.function_extender import ExtenderHook, Extender


class _FakeExtender(Extender):
    """Minimal concrete Extender double, mirrors MockExtender in test_composite_extender.py."""

    def __init__(self, name: str, priority: int = 100) -> None:
        self.name = name
        self.priority = priority
        self.call_count = 0

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return func(*args, **kwargs)


def test_composite_extender_importable_from_function_extender_module() -> None:
    """CompositeExtender (no underscore) must be importable directly from function_extender."""
    from mloda.core.abstract_plugins.function_extender import CompositeExtender

    assert CompositeExtender is not None


def test_composite_extender_importable_from_steward() -> None:
    """CompositeExtender must be re-exported from the public mloda.steward package."""
    from mloda.steward import CompositeExtender

    assert CompositeExtender is not None


def test_composite_extender_listed_in_steward_all() -> None:
    """CompositeExtender must be listed in mloda.steward.__all__."""
    import mloda.steward as steward

    assert "CompositeExtender" in steward.__all__


def test_old_private_name_no_longer_exists() -> None:
    """The rename drops the old private name; it must not remain as an alias."""
    assert not hasattr(function_extender_module, "_CompositeExtender"), (
        "_CompositeExtender must be renamed to CompositeExtender, not aliased"
    )


def test_composite_extender_still_chains_and_aggregates() -> None:
    """The renamed public CompositeExtender still chains extenders and unions wraps()."""
    from mloda.core.abstract_plugins.function_extender import CompositeExtender

    extender1 = _FakeExtender("first", priority=10)
    extender2 = _FakeExtender("second", priority=20)
    composite = CompositeExtender([extender1, extender2])

    wrapped = composite.wraps()
    assert ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE in wrapped

    def add(x: int, y: int) -> int:
        return x + y

    result = composite(add, 5, 3)

    assert result == 8
    assert extender1.call_count == 1
    assert extender2.call_count == 1
