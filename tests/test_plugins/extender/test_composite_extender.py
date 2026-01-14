"""Tests for composite extender functionality allowing multiple Extenders.

This module tests the ability to have multiple Extender instances
wrapping the same function type, with priority-based execution order and error resilience.
"""

from typing import Any, Callable, Set
from unittest.mock import Mock
import pytest
import logging

from mloda.core.abstract_plugins.function_extender import (
    ExtenderHook,
    Extender,
)
from mloda_plugins.function_extender.base_implementations.otel.otel_extender import OtelExtender
from mloda.core.abstract_plugins.function_extender import _CompositeExtender
from mloda.provider import ComputeFramework


class MockExtender(Extender):
    """Mock extender for testing purposes."""

    def __init__(self, name: str, priority: int = 100, should_fail: bool = False):
        self.name = name
        self.priority = priority
        self.should_fail = should_fail
        self.call_count = 0

    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        if self.should_fail:
            raise ValueError(f"MockExtender {self.name} intentionally failed")
        result = func(*args, **kwargs)
        return result


class TestExtenderPriority:
    """Test that Extender has a priority property."""

    def test_priority_property_exists_on_base_class(self) -> None:
        """Test that Extender base class defines priority."""
        # Check if priority is defined at the class level or in __init__
        # OtelExtender (a real implementation) should have priority after implementation

        otel = OtelExtender()
        assert hasattr(otel, "priority"), "Extender implementations must have a priority property"

    def test_priority_default_value(self) -> None:
        """Test that priority defaults to 100 when not specified."""
        otel = OtelExtender()
        assert otel.priority == 100, "Default priority should be 100"

    def test_priority_custom_value(self) -> None:
        """Test that priority can be set to a custom value."""

        # This will fail until priority can be customized in __init__
        class CustomPriorityExtender(Extender):
            def __init__(self, priority: int = 100):
                self.priority = priority

            def wraps(self) -> Set[ExtenderHook]:
                return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

            def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

        extender = CustomPriorityExtender(priority=50)
        assert extender.priority == 50, "Priority should be settable to custom value"


class Test_CompositeExtender:
    """Test _CompositeExtender class that chains multiple extenders."""

    def test_composite_extender_class_exists(self) -> None:
        """Test that _CompositeExtender class is defined."""
        # This will fail until _CompositeExtender is implemented
        assert _CompositeExtender is not None, "_CompositeExtender class must exist"

    def test_composite_extender_inherits_from_wrapper(self) -> None:
        """Test that _CompositeExtender inherits from Extender."""

        # This will fail until _CompositeExtender inherits properly
        assert issubclass(_CompositeExtender, Extender), "_CompositeExtender must inherit from Extender"

    def test_composite_extender_accepts_list_of_extenders(self) -> None:
        """Test that _CompositeExtender can be initialized with a list of extenders."""

        extender1 = MockExtender("first", priority=10)
        extender2 = MockExtender("second", priority=20)

        # This will fail until _CompositeExtender accepts extenders in __init__
        composite = _CompositeExtender([extender1, extender2])
        assert composite is not None, "_CompositeExtender should accept list of extenders"

    def test_composite_extender_chains_multiple_extenders(self) -> None:
        """Test that _CompositeExtender calls all extenders in the chain."""

        extender1 = MockExtender("first", priority=10)
        extender2 = MockExtender("second", priority=20)
        composite = _CompositeExtender([extender1, extender2])

        def test_func(x: int, y: int) -> int:
            return x + y

        # This will fail until _CompositeExtender calls all extenders
        result = composite(test_func, 5, 3)

        assert result == 8, "Function should execute correctly"
        assert extender1.call_count == 1, "First extender should be called"
        assert extender2.call_count == 1, "Second extender should be called"

    def test_composite_extender_wraps_returns_union(self) -> None:
        """Test that _CompositeExtender.wraps() returns union of all wrapped types."""

        extender1 = MockExtender("first")
        extender2 = MockExtender("second")
        composite = _CompositeExtender([extender1, extender2])

        # This will fail until _CompositeExtender.wraps() returns proper union
        wrapped = composite.wraps()
        assert ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE in wrapped, (
            "_CompositeExtender should wrap all function types from child extenders"
        )


class TestExtenderExecutionOrder:
    """Test that extenders execute in priority order (lower first)."""

    def test_extenders_execute_in_priority_order(self) -> None:
        """Test that extenders with lower priority execute first."""
        execution_order = []

        class OrderTrackingExtender(Extender):
            def __init__(self, name: str, priority: int):
                self.name = name
                self.priority = priority

            def wraps(self) -> Set[ExtenderHook]:
                return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

            def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
                execution_order.append(self.name)
                return func(*args, **kwargs)

        # Create extenders with different priorities (intentionally out of order)
        extender_high = OrderTrackingExtender("high_priority_50", priority=50)
        extender_low = OrderTrackingExtender("low_priority_10", priority=10)
        extender_mid = OrderTrackingExtender("mid_priority_30", priority=30)

        # This will fail until _CompositeExtender sorts by priority
        composite = _CompositeExtender([extender_high, extender_low, extender_mid])

        def test_func() -> str:
            return "done"

        composite(test_func)

        # Should execute in order: low (10), mid (30), high (50)
        assert execution_order == ["low_priority_10", "mid_priority_30", "high_priority_50"], (
            "Extenders should execute in priority order (lower priority value first)"
        )


class TestExtenderErrorResilience:
    """Test that if one extender fails, the chain continues."""

    def test_continues_after_extender_failure(self) -> None:
        """Test that if one extender raises an exception, remaining extenders still execute."""

        extender1 = MockExtender("first", priority=10)
        extender2 = MockExtender("failing", priority=20, should_fail=True)
        extender3 = MockExtender("third", priority=30)

        composite = _CompositeExtender([extender1, extender2, extender3])

        def test_func(x: int) -> int:
            return x * 2

        # This will fail until _CompositeExtender handles exceptions gracefully
        result = composite(test_func, 5)

        assert result == 10, "Function should still execute despite middle extender failure"
        assert extender1.call_count == 1, "First extender should execute"
        assert extender2.call_count == 1, "Failing extender should be attempted"
        assert extender3.call_count == 1, "Third extender should execute despite failure"

    def test_logs_error_when_extender_fails(self, caplog: Any) -> None:
        """Test that extender failures are logged."""

        extender1 = MockExtender("first", priority=10)
        extender2 = MockExtender("failing", priority=20, should_fail=True)

        composite = _CompositeExtender([extender1, extender2])

        def test_func(x: int) -> int:
            return x * 2

        with caplog.at_level(logging.ERROR):
            # This will fail until _CompositeExtender logs errors
            composite(test_func, 5)

        # Check that error was logged
        assert any("MockExtender failing intentionally failed" in record.message for record in caplog.records), (
            "Extender failures should be logged"
        )

    def test_original_function_error_propagates(self) -> None:
        """Test that errors from the original function are not caught."""

        extender = MockExtender("test", priority=10)
        composite = _CompositeExtender([extender])

        def failing_func() -> None:
            raise RuntimeError("Original function error")

        # This will fail until error handling only catches extender errors, not function errors
        with pytest.raises(RuntimeError, match="Original function error"):
            composite(failing_func)


class TestGetFunctionExtenderWithComposite:
    """Test that get_function_extender returns _CompositeExtender for multiple matches."""

    def test_get_function_extender_returns_composite_for_multiple_matches(self) -> None:
        """Test that get_function_extender returns a _CompositeExtender when multiple extenders match."""

        # Create a mock ComputeFramework with multiple extenders
        extender1 = MockExtender("first", priority=10)
        extender2 = MockExtender("second", priority=20)

        # This will fail until get_function_extender supports multiple extenders
        # Currently it raises ValueError for multiple matches
        compute_fw = Mock(spec=ComputeFramework)
        compute_fw.function_extender = [extender1, extender2]
        compute_fw.get_function_extender = ComputeFramework.get_function_extender.__get__(compute_fw)

        result = compute_fw.get_function_extender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)

        assert isinstance(result, _CompositeExtender), (
            "get_function_extender should return _CompositeExtender for multiple matches"
        )
        assert isinstance(result, Extender), "_CompositeExtender should be a Extender"

    def test_get_function_extender_preserves_priority_order(self) -> None:
        """Test that get_function_extender creates _CompositeExtender with correct priority order."""

        # Create extenders in non-priority order
        extender_high = MockExtender("high", priority=50)
        extender_low = MockExtender("low", priority=10)
        extender_mid = MockExtender("mid", priority=30)

        compute_fw = Mock(spec=ComputeFramework)
        compute_fw.function_extender = [extender_high, extender_low, extender_mid]
        compute_fw.get_function_extender = ComputeFramework.get_function_extender.__get__(compute_fw)

        # This will fail until _CompositeExtender sorts extenders by priority
        result = compute_fw.get_function_extender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)

        assert isinstance(result, _CompositeExtender), "Should return _CompositeExtender"

        # Execute and verify order
        execution_order = []

        def track_order(name: str) -> Callable[..., Any]:
            def inner(func: Any, *args: Any, **kwargs: Any) -> Any:
                execution_order.append(name)
                return func(*args, **kwargs)

            return inner

        # Patch the extender calls to track order
        extender_low.__call__ = track_order("low")  # type: ignore[method-assign]
        extender_mid.__call__ = track_order("mid")  # type: ignore[method-assign]
        extender_high.__call__ = track_order("high")  # type: ignore[method-assign]

        def test_func() -> str:
            return "done"

        result(test_func)

        assert execution_order == ["low", "mid", "high"], "_CompositeExtender should maintain priority order"

    def test_get_function_extender_single_match_unchanged(self) -> None:
        """Test that get_function_extender returns single extender directly when only one matches."""
        extender = MockExtender("only", priority=10)

        compute_fw = Mock(spec=ComputeFramework)
        compute_fw.function_extender = [extender]
        compute_fw.get_function_extender = ComputeFramework.get_function_extender.__get__(compute_fw)

        # This should continue to work as before - single extender returned directly
        result = compute_fw.get_function_extender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)

        assert result is extender, "Single matching extender should be returned directly (backward compatibility)"
