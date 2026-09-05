"""Tests for HookContext, the delivery seam handed to Extender implementations.

Pins construction, the ambient current()/activate() scope (including nested
restore), row_count's __len__ gating, and instrument's timing/status bookkeeping.
"""

from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext, instrument


class _NoLenDouble:
    """Stand-in for a lazy frame (e.g. polars LazyFrame) that deliberately lacks __len__."""


def _make_context(**overrides: Any) -> HookContext:
    required = {
        "hook": ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
        "feature_group_class": "tests.something.FakeFeatureGroup",
        "feature_group_version": "v1",
        "plugin_version": None,
        "feature_names": ("my_feature",),
        "input_features": None,
        "compute_framework_name": "FakeFramework",
    }
    required.update(overrides)
    return HookContext(**required)  # type: ignore[arg-type]


class TestHookContextConstruction:
    """HookContext construction and defaults."""

    def test_constructs_with_only_required_fields(self) -> None:
        context = _make_context()

        assert context.hook == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
        assert context.feature_group_class == "tests.something.FakeFeatureGroup"
        assert context.feature_group_version == "v1"
        assert context.plugin_version is None
        assert context.feature_names == ("my_feature",)
        assert context.input_features is None
        assert context.compute_framework_name == "FakeFramework"

    def test_defaulted_fields_are_none(self) -> None:
        context = _make_context()

        assert context.rows_in is None
        assert context.rows_out is None
        assert context.duration_seconds is None
        assert context.status is None
        assert context.run_id is None
        assert context.data_access_identity is None
        assert context.tenant_id is None
        assert context.project_id is None
        assert context.principal is None


class TestHookContextCarrierField:
    """HookContext carries an opaque W3C trace-context carrier dict end to end."""

    def test_carrier_defaults_to_none(self) -> None:
        context = _make_context()

        assert context.carrier is None

    def test_carrier_can_be_set_via_constructor(self) -> None:
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}

        context = _make_context(carrier=carrier)

        assert context.carrier == carrier


class TestHookContextWorkerIndexField:
    """HookContext carries the worker index of the multiprocessing worker it ran in."""

    def test_worker_index_defaults_to_none(self) -> None:
        context = _make_context()

        assert context.worker_index is None

    def test_worker_index_can_be_set_via_constructor(self) -> None:
        context = _make_context(worker_index=3)

        assert context.worker_index == 3


class TestHookContextDataAccessAndJoinAndPlanFields:
    """Optional fields for the FEATURE_GROUP_MATCHED/INPUT_DATA_LOAD/JOIN hooks."""

    def test_new_fields_default_to_none(self) -> None:
        context = _make_context()

        assert context.data_access_format is None
        assert context.data_access_dataset_version is None
        assert context.join_type is None
        assert context.join_keys is None
        assert context.plan_feature_count is None
        assert context.plan_node_count is None
        assert context.plan_depth is None

    def test_new_fields_can_be_set_via_constructor(self) -> None:
        context = _make_context(
            data_access_format="parquet",
            data_access_dataset_version="2024-01-01",
            join_type="inner",
            join_keys=("id", "date"),
            plan_feature_count=5,
            plan_node_count=12,
            plan_depth=3,
        )

        assert context.data_access_format == "parquet"
        assert context.data_access_dataset_version == "2024-01-01"
        assert context.join_type == "inner"
        assert context.join_keys == ("id", "date")
        assert context.plan_feature_count == 5
        assert context.plan_node_count == 12
        assert context.plan_depth == 3


class TestHookContextCurrentScope:
    """HookContext.current() reflects the active activate() scope, with proper nested restore."""

    def test_current_returns_none_when_no_scope_active(self) -> None:
        assert HookContext.current() is None

    def test_current_returns_active_context_inside_activate(self) -> None:
        context = _make_context()

        with context.activate():
            assert HookContext.current() is context

    def test_current_returns_none_after_activate_exits(self) -> None:
        context = _make_context()

        with context.activate():
            pass

        assert HookContext.current() is None

    def test_nested_activate_restores_outer_context_not_none(self) -> None:
        outer = _make_context()
        inner = _make_context(feature_names=("inner_feature",))

        with outer.activate():
            assert HookContext.current() is outer
            with inner.activate():
                assert HookContext.current() is inner
            assert HookContext.current() is outer, "Exiting the inner scope must restore the outer context"

        assert HookContext.current() is None


class TestHookContextOutputSchemaField:
    """HookContext carries a best-effort output schema: (column_name, dtype_or_none) pairs."""

    def test_defaults_to_none(self) -> None:
        context = _make_context()

        assert context.output_schema is None

    def test_settable_via_constructor(self) -> None:
        context = _make_context(output_schema=(("a", "int64"), ("b", None)))

        assert context.output_schema == (("a", "int64"), ("b", None))


class TestHookContextRowCount:
    """HookContext.row_count is __len__-gated, never calls len() on unsized objects."""

    def test_returns_length_for_sized_objects(self) -> None:
        assert HookContext.row_count([1, 2, 3]) == 3
        assert HookContext.row_count("abcde") == 5

    def test_returns_none_without_calling_len_on_object_without_len(self) -> None:
        assert HookContext.row_count(object()) is None

        lazy_frame_double = _NoLenDouble()
        assert not hasattr(lazy_frame_double, "__len__")
        assert HookContext.row_count(lazy_frame_double) is None


class TestHookContextRowCountNonColumnarDicts:
    """A dict's first column must be row-shaped (list/tuple-like); scalar, nested-dict, or bytes values return None."""

    def test_scalar_string_first_value_returns_none(self) -> None:
        assert HookContext.row_count({"name": "hello"}) is None

    def test_nested_dict_first_value_returns_none(self) -> None:
        assert HookContext.row_count({"a": {"x": [1, 2, 3]}}) is None

    def test_bytes_first_value_returns_none(self) -> None:
        assert HookContext.row_count({"a": b"bytes"}) is None

    def test_list_first_value_still_counts(self) -> None:
        assert HookContext.row_count({"a": [1, 2, 3]}) == 3

    def test_tuple_first_value_still_counts(self) -> None:
        assert HookContext.row_count({"a": (1, 2)}) == 2

    def test_empty_dict_returns_zero(self) -> None:
        assert HookContext.row_count({}) == 0


class TestInstrument:
    """instrument wraps a callable, updating context.status/duration_seconds/rows_out."""

    def test_wrapper_returns_same_value_as_direct_call(self) -> None:
        context = _make_context()

        def raw(x: int, y: int) -> int:
            return x + y

        wrapped = instrument(context, raw)
        assert wrapped(2, 3) == raw(2, 3) == 5

    def test_sets_status_success_after_successful_call(self) -> None:
        context = _make_context()

        def raw() -> str:
            return "ok"

        wrapped = instrument(context, raw)
        wrapped()

        assert context.status == "success"

    def test_sets_duration_seconds_on_success(self) -> None:
        context = _make_context()

        def raw() -> str:
            return "ok"

        wrapped = instrument(context, raw)
        wrapped()

        assert context.duration_seconds is not None
        assert context.duration_seconds >= 0

    def test_sets_rows_out_for_list_result(self) -> None:
        context = _make_context()

        def raw() -> list[int]:
            return [1, 2, 3, 4]

        wrapped = instrument(context, raw)
        result = wrapped()

        assert context.rows_out == HookContext.row_count(result) == 4

    def test_forwards_args_and_kwargs_unchanged(self) -> None:
        received: dict[str, Any] = {}

        def raw(a: int, b: int, *, c: str) -> str:
            received["args"] = (a, b)
            received["kwargs"] = {"c": c}
            return c

        context = _make_context()
        wrapped = instrument(context, raw)
        result = wrapped(1, 2, c="three")

        assert received["args"] == (1, 2)
        assert received["kwargs"] == {"c": "three"}
        assert result == "three"

    def test_propagates_exception_and_sets_status_error(self) -> None:
        context = _make_context()

        def raw() -> None:
            raise ValueError("boom")

        wrapped = instrument(context, raw)

        with pytest.raises(ValueError, match="boom"):
            wrapped()

        assert context.status == "error"
        assert context.duration_seconds is not None

    def test_calls_wrapped_function_exactly_once_on_success(self) -> None:
        call_count = {"n": 0}
        context = _make_context()

        def raw() -> str:
            call_count["n"] += 1
            return "ok"

        wrapped = instrument(context, raw)
        wrapped()

        assert call_count["n"] == 1

    def test_calls_wrapped_function_exactly_once_on_error(self) -> None:
        call_count = {"n": 0}
        context = _make_context()

        def raw() -> None:
            call_count["n"] += 1
            raise RuntimeError("boom")

        wrapped = instrument(context, raw)

        with pytest.raises(RuntimeError, match="boom"):
            wrapped()

        assert call_count["n"] == 1


class TestInstrumentOutputSchema:
    """instrument's optional output_schema kwarg mirrors rows_out's reset/degrade semantics."""

    def test_default_leaves_output_schema_none_after_success(self) -> None:
        context = _make_context()

        def raw() -> list[int]:
            return [1, 2, 3]

        wrapped = instrument(context, raw)
        wrapped()

        assert context.output_schema is None

    def test_callable_result_sets_output_schema(self) -> None:
        context = _make_context()

        def raw() -> list[int]:
            return [1, 2, 3]

        wrapped = instrument(context, raw, output_schema=lambda result: (("a", "int64"),))
        wrapped()

        assert context.output_schema == (("a", "int64"),)

    def test_callable_receives_wrapped_function_return_value(self) -> None:
        context = _make_context()
        received: dict[str, Any] = {}

        def raw() -> list[int]:
            return [1, 2, 3]

        def capture_schema(result: Any) -> Any:
            received["result"] = result
            return None

        wrapped = instrument(context, raw, output_schema=capture_schema)
        result = wrapped()

        assert received["result"] is result

    def test_raising_output_schema_callable_degrades_to_none_without_propagating(self) -> None:
        context = _make_context()

        def raw() -> list[int]:
            return [1, 2, 3]

        def raising_schema(result: Any) -> Any:
            raise RuntimeError("boom")

        wrapped = instrument(context, raw, output_schema=raising_schema)
        result = wrapped()

        assert result == [1, 2, 3]
        assert context.output_schema is None

    def test_reset_at_call_start_and_stays_none_after_raise(self) -> None:
        context = _make_context(output_schema=(("stale", "int"),))
        recorded: dict[str, Any] = {}

        def raw() -> None:
            recorded["output_schema_during_call"] = context.output_schema
            raise ValueError("boom")

        wrapped = instrument(context, raw, output_schema=lambda result: (("a", "int64"),))

        with pytest.raises(ValueError, match="boom"):
            wrapped()

        assert recorded["output_schema_during_call"] is None
        assert context.output_schema is None


class TestInstrumentStatusStaysNoneUntilCallFinishes:
    """instrument must not pre-set status to 'error'; status stays None while func is running."""

    def test_status_is_none_during_the_call_then_success_after(self) -> None:
        context = _make_context()
        recorded: dict[str, Any] = {}

        def raw() -> list[int]:
            recorded["status_during_call"] = context.status
            return [1]

        wrapped = instrument(context, raw)
        wrapped()

        assert recorded["status_during_call"] is None
        assert context.status == "success"

    def test_status_is_error_after_the_call_raises(self) -> None:
        context = _make_context()

        def raw() -> None:
            raise ValueError("boom")

        wrapped = instrument(context, raw)

        with pytest.raises(ValueError, match="boom"):
            wrapped()

        assert context.status == "error"


class _SelfCarryingFeatureGroup:
    """FeatureGroup-like class exposing calculate_feature as a bound classmethod."""

    @classmethod
    def calculate_feature(cls, data: Any) -> Any:
        return data


class TestInstrumentPreservesSelf:
    """instrument's wrapper must copy __self__ from the wrapped callable when present."""

    def test_wrapper_carries_self_from_bound_classmethod(self) -> None:
        context = _make_context()
        bound = _SelfCarryingFeatureGroup.calculate_feature

        wrapped = instrument(context, bound)

        assert hasattr(wrapped, "__self__")
        assert wrapped.__self__ is bound.__self__  # type: ignore[attr-defined]

    def test_wrapper_has_no_self_for_plain_function_without_one(self) -> None:
        context = _make_context()

        def plain(x: int) -> int:
            return x

        wrapped = instrument(context, plain)

        assert not hasattr(wrapped, "__self__")
