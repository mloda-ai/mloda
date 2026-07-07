"""Tests for composite extender functionality allowing multiple Extenders.

This module tests the ability to have multiple Extender instances
wrapping the same function type, with priority-based execution order and error resilience.
"""

from typing import Any, Callable
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
    """Mock extender for testing purposes.

    ``raise_on_error`` mirrors the new error contract: when True (default) an
    extender failure must break the calculation; when False the extender opts
    into warning-only behavior (log + fall back to the wrapped function).
    """

    def __init__(
        self,
        name: str,
        priority: int = 100,
        should_fail: bool = False,
        raise_on_error: bool = True,
    ):
        self.name = name
        self.priority = priority
        self.should_fail = should_fail
        # Uses the base-class property/setter (same pattern as ``priority``)
        # so the mock reports its opt-in via ``raise_on_error``.
        self.raise_on_error = raise_on_error
        self.call_count = 0

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        if self.should_fail:
            raise ValueError(f"MockExtender {self.name} intentionally failed")
        result = func(*args, **kwargs)
        return result


class _PostFailExtender(Extender):
    """Warning-only extender that delegates successfully, then fails in its OWN post-processing.

    ``__call__`` runs the wrapped function first (so the inner result is already
    computed) and only afterwards raises. The correct fallback must return the
    already-computed inner result WITHOUT re-running the wrapped function.
    """

    def __init__(self, name: str, priority: int = 100) -> None:
        self.name = name
        self.priority = priority
        self.raise_on_error = False

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        func(*args, **kwargs)
        raise ValueError("post boom")


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

            def wraps(self) -> set[ExtenderHook]:
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

            def wraps(self) -> set[ExtenderHook]:
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
        """A WARNING-ONLY extender that fails must not break the chain.

        Under the new contract, only extenders that opt out of breaking
        (``raise_on_error=False``) fall back; the rest of the chain and the
        wrapped function still execute.
        """

        extender1 = MockExtender("first", priority=10)
        extender2 = MockExtender("failing", priority=20, should_fail=True, raise_on_error=False)
        extender3 = MockExtender("third", priority=30)

        composite = _CompositeExtender([extender1, extender2, extender3])

        def test_func(x: int) -> int:
            return x * 2

        result = composite(test_func, 5)

        assert result == 10, "Function should still execute despite middle warning-only extender failure"
        assert extender1.call_count == 1, "First extender should execute"
        assert extender2.call_count == 1, "Failing extender should be attempted"
        assert extender3.call_count == 1, "Third extender should execute despite failure"

    def test_logs_error_when_extender_fails(self, caplog: Any) -> None:
        """A warning-only extender failure is logged at WARNING level."""

        extender1 = MockExtender("first", priority=10)
        extender2 = MockExtender("failing", priority=20, should_fail=True, raise_on_error=False)

        composite = _CompositeExtender([extender1, extender2])

        def test_func(x: int) -> int:
            return x * 2

        with caplog.at_level(logging.WARNING):
            composite(test_func, 5)

        assert any(
            record.levelno == logging.WARNING and "MockExtender failing intentionally failed" in record.message
            for record in caplog.records
        ), "Warning-only extender failures should be logged at WARNING level and identify the failure"

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


class TestExtenderRaiseOnErrorProperty:
    """Test the new ``raise_on_error`` property (mirrors the ``priority`` property)."""

    def test_raise_on_error_default_true(self) -> None:
        """raise_on_error defaults to True on a plain Extender subclass (breaking by default)."""

        class PlainExtender(Extender):
            def wraps(self) -> set[ExtenderHook]:
                return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

            def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

        extender = PlainExtender()
        assert extender.raise_on_error is True, "Extenders must break by default (raise_on_error=True)"

    def test_raise_on_error_settable_to_false(self) -> None:
        """raise_on_error can be set to False to opt into warning-only behavior."""

        class PlainExtender(Extender):
            def wraps(self) -> set[ExtenderHook]:
                return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

            def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

        extender = PlainExtender()
        extender.raise_on_error = False
        assert extender.raise_on_error is False, "raise_on_error should be settable to False"


class TestExtenderRaiseOnErrorContractComposite:
    """Pin the new breaking-by-default error contract for the composite path."""

    def test_default_extender_failure_re_raises(self) -> None:
        """A DEFAULT (raise_on_error=True) extender that fails must re-raise, not swallow/fall back."""

        failing = MockExtender("boom", priority=10, should_fail=True)  # raise_on_error defaults True
        composite = _CompositeExtender([failing])

        def test_func(x: int) -> int:
            return x * 2

        with pytest.raises(ValueError, match="MockExtender boom intentionally failed"):
            composite(test_func, 5)

    def test_default_extender_failure_does_not_call_fallback(self) -> None:
        """A breaking extender failure must not silently run the wrapped function as a fallback."""

        call_marker = {"func_calls": 0}

        failing = MockExtender("boom", priority=10, should_fail=True)
        composite = _CompositeExtender([failing])

        def test_func(x: int) -> int:
            call_marker["func_calls"] += 1
            return x * 2

        with pytest.raises(ValueError, match="MockExtender boom intentionally failed"):
            composite(test_func, 5)

        assert call_marker["func_calls"] == 0, "Breaking extender must not fall back to the wrapped function"

    def test_adding_extender_does_not_change_error_semantics(self) -> None:
        """SYMMETRY: a single breaking extender raises; adding a second breaking extender still raises."""

        def test_func(x: int) -> int:
            return x * 2

        # Single breaking extender
        single = _CompositeExtender([MockExtender("boom", priority=10, should_fail=True)])
        with pytest.raises(ValueError, match="MockExtender boom intentionally failed"):
            single(test_func, 5)

        # Adding a second (non-failing) breaking extender must not change the semantics
        composite = _CompositeExtender(
            [
                MockExtender("boom", priority=10, should_fail=True),
                MockExtender("ok", priority=20),
            ]
        )
        with pytest.raises(ValueError, match="MockExtender boom intentionally failed"):
            composite(test_func, 5)

    def test_warning_only_extender_falls_back_and_warns(self, caplog: Any) -> None:
        """A warning-only (raise_on_error=False) extender that fails logs a warning and falls back."""

        failing = MockExtender("soft", priority=10, should_fail=True, raise_on_error=False)
        other = MockExtender("ok", priority=20)
        composite = _CompositeExtender([failing, other])

        def test_func(x: int) -> int:
            return x * 2

        with caplog.at_level(logging.WARNING):
            result = composite(test_func, 5)

        assert result == 10, "Warning-only extender failure should fall back to the wrapped function"
        assert other.call_count == 1, "Remaining extenders should still execute"
        assert any(
            record.levelno == logging.WARNING and "MockExtender soft intentionally failed" in record.message
            for record in caplog.records
        ), "Warning-only failure should be logged at WARNING level"


class _FakeFeatureGroup:
    """Minimal feature group exercising the single-extender path of run_calculate_feature."""

    def __init__(self) -> None:
        self.calc_count = 0

    def calculate_feature(self, data: Any, features: Any) -> Any:
        self.calc_count += 1
        return f"calculated:{data}"

    def get_class_name(self) -> str:
        return "FakeFeatureGroup"


class _KeyErrorFeatureGroup:
    """Feature group whose calculate_feature raises KeyError, counting invocations."""

    def __init__(self) -> None:
        self.calc_count = 0

    def calculate_feature(self, data: Any, features: Any) -> Any:
        self.calc_count += 1
        raise KeyError("missing_col")

    def get_class_name(self) -> str:
        return "KeyErrorFeatureGroup"


class _RuntimeErrorFeatureGroup:
    """Feature group whose calculate_feature raises RuntimeError, counting invocations."""

    def __init__(self) -> None:
        self.calc_count = 0

    def calculate_feature(self, data: Any, features: Any) -> Any:
        self.calc_count += 1
        raise RuntimeError("inner boom")

    def get_class_name(self) -> str:
        return "RuntimeErrorFeatureGroup"


class TestWarningOnlyDoesNotSwallowInnerErrors:
    """Pin that a warning-only extender only handles its OWN failures.

    Exceptions raised by the inner function or a downstream breaking extender must
    propagate unchanged, and the inner function must never be executed twice.
    """

    def test_warning_only_outer_does_not_swallow_breaking_inner(self, caplog: Any) -> None:
        """COMPOSITE A: warning-only OUTER must let a breaking INNER extender's exception propagate.

        The breaking extender (raise_on_error=True) runs inner to the warning-only
        extender. When it raises, the outer warning-only fallback must NOT catch it,
        must NOT re-run the breaking extender, and must NOT log a spurious warning.
        """

        base_calls = {"n": 0}

        def base(x: int) -> int:
            base_calls["n"] += 1
            return x * 2

        warn = MockExtender("warn", priority=10, raise_on_error=False)
        breaking = MockExtender("boom", priority=20, should_fail=True, raise_on_error=True)
        composite = _CompositeExtender([warn, breaking])

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError, match="MockExtender boom intentionally failed"):
                composite(base, 5)

        assert breaking.call_count == 1, (
            "Breaking inner extender must run exactly once; the outer warning-only fallback must not re-run it"
        )
        assert base_calls["n"] == 0, "Base function must not run when the breaking extender raises before delegating"
        assert not [r for r in caplog.records if r.levelno == logging.WARNING], (
            "Warning-only outer extender must not log a spurious warning for the breaking inner extender's failure"
        )

    def test_warning_only_does_not_swallow_inner_function_error(self) -> None:
        """COMPOSITE B: an inner-function exception through a warning-only extender propagates, single run."""

        base_calls = {"n": 0}

        def base() -> int:
            base_calls["n"] += 1
            raise RuntimeError("inner boom")

        warn = MockExtender("warn", priority=10, raise_on_error=False)
        composite = _CompositeExtender([warn])

        with pytest.raises(RuntimeError, match="inner boom"):
            composite(base)

        assert base_calls["n"] == 1, (
            "The inner function must run exactly once; the warning-only wrapper must not swallow and re-run it"
        )

    def test_warning_only_post_failure_returns_computed_result_without_rerun(self, caplog: Any) -> None:
        """COMPOSITE C: extender fails AFTER inner ran, must return already-computed result, no double run."""

        base_calls = {"n": 0}

        def base(x: int) -> int:
            base_calls["n"] += 1
            return x * 3

        post_fail = _PostFailExtender("posty", priority=10)
        composite = _CompositeExtender([post_fail])

        with caplog.at_level(logging.WARNING):
            result = composite(base, 7)

        assert result == 21, "Fallback must return the already-computed inner result"
        assert base_calls["n"] == 1, (
            "Inner function must run exactly once; no double execution after the extender's post-processing fails"
        )
        assert any(
            r.levelno == logging.WARNING and "post boom" in r.message and "posty" in r.message for r in caplog.records
        ), "The failing extender must be identified in a WARNING"


class TestSingleExtenderPathHonorsRaiseOnError:
    """The single-extender path (run_calculate_feature) must honor raise_on_error identically."""

    @staticmethod
    def _make_compute_framework(extenders: list[Extender]) -> Any:
        cf = Mock(spec=ComputeFramework)
        cf.data = "DATA"
        cf.function_extender = extenders
        cf.get_function_extender = ComputeFramework.get_function_extender.__get__(cf)
        cf.run_calculate_feature = ComputeFramework.run_calculate_feature.__get__(cf)
        cf._raise_helpful_missing_column_error = ComputeFramework._raise_helpful_missing_column_error.__get__(cf)
        return cf

    def test_single_default_extender_failure_propagates(self) -> None:
        """A single DEFAULT extender that fails must propagate (symmetry with composite)."""

        cf = self._make_compute_framework([MockExtender("boom", should_fail=True)])
        fg = _FakeFeatureGroup()

        with pytest.raises(ValueError, match="MockExtender boom intentionally failed"):
            cf.run_calculate_feature(fg, "features")

        assert fg.calc_count == 0, "Breaking extender must not fall back to calculate_feature"

    def test_single_warning_only_extender_falls_back(self, caplog: Any) -> None:
        """A single WARNING-ONLY extender that fails logs a warning and falls back to calculate_feature."""

        cf = self._make_compute_framework([MockExtender("soft", should_fail=True, raise_on_error=False)])
        fg = _FakeFeatureGroup()

        with caplog.at_level(logging.WARNING):
            result = cf.run_calculate_feature(fg, "features")

        assert result == "calculated:DATA", "Warning-only extender should fall back to calculate_feature"
        assert fg.calc_count == 1, "The wrapped calculate_feature should run exactly once as fallback"
        assert any(
            record.levelno == logging.WARNING and "MockExtender soft intentionally failed" in record.message
            for record in caplog.records
        ), "Warning-only failure should be logged at WARNING level on the single-extender path"

    def test_single_warning_only_inner_keyerror_propagates_once(self) -> None:
        """SINGLE D: an inner KeyError through a warning-only extender must not be swallowed/re-run.

        The KeyError must propagate through the same handling a plain (no-extender)
        KeyError would receive (the helpful ValueError), and calculate_feature must
        run exactly once.
        """

        cf = self._make_compute_framework([MockExtender("soft", should_fail=False, raise_on_error=False)])
        fg = _KeyErrorFeatureGroup()

        with pytest.raises(ValueError, match="failed with a KeyError"):
            cf.run_calculate_feature(fg, "features")

        assert fg.calc_count == 1, (
            "calculate_feature must run exactly once; the warning-only fallback must not re-run it on inner KeyError"
        )

    def test_single_warning_only_inner_runtime_error_propagates_once(self) -> None:
        """SINGLE (analogue of B): an inner RuntimeError through a warning-only extender propagates, single run."""

        cf = self._make_compute_framework([MockExtender("soft", should_fail=False, raise_on_error=False)])
        fg = _RuntimeErrorFeatureGroup()

        with pytest.raises(RuntimeError, match="inner boom"):
            cf.run_calculate_feature(fg, "features")

        assert fg.calc_count == 1, (
            "calculate_feature must run exactly once on inner RuntimeError through a warning-only extender"
        )
