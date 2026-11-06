"""Conformance suites for the empty-result schema contract and the ComputeFramework._append_columns hook."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

from mloda._testing.compute_framework.host import ComputeFrameworkSuite
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.source import reclaim
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import DataCreatorFeatureGroup
from mloda.user import DataType

_ONE_ROW = TypedTestData(
    {"a": (DataType.INT64, (1,)), "flag": (DataType.BOOLEAN, (True,)), "name": (DataType.STRING, ("x",))}
)
_EMPTY = TypedTestData({"a": (DataType.INT64, ()), "flag": (DataType.BOOLEAN, ()), "name": (DataType.STRING, ())})
_APPENDABLE = TypedTestData(
    {
        "a": (DataType.INT64, (1, 2, 3)),
        "n": (DataType.INT64, (1, None, 3)),
        "flag": (DataType.BOOLEAN, (True, None, False)),
    }
)


def _assert_type(adapter: FrameworkAdapter, frame: Any, column: str, declared: DataType, reference: Any) -> None:
    """has_type must not be False; where the adapter cannot tell (None), the type must equal the reference frame's."""
    exact = adapter.has_type(frame, column, declared)
    if exact is None:
        assert adapter.column_type(frame, column) == adapter.column_type(reference, column), (
            f"column '{column}' changed type from {adapter.column_type(reference, column)}"
        )
    else:
        assert exact, f"column '{column}' lost the declared type {declared}"


class EmptyResultSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of extract_column_names on zero-row and non-empty frames; every framework is served."""

    aspect = "empty result"

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return True

    def test_extract_column_names_on_empty_data(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        columns = framework_instance._extract_column_names(self.frame(adapter, "empty", _EMPTY))
        assert columns, "schema-bearing zero-row frame must expose its columns (state C)"

    def test_empty_frame_keeps_declared_column_types(self, adapter: FrameworkAdapter) -> None:
        empty = self.frame(adapter, "empty", _EMPTY)
        reference = self.frame(adapter, "one_row", _ONE_ROW)
        for column, declared in _EMPTY.types.items():
            _assert_type(adapter, empty, column, declared, reference)

    def test_extract_column_names_on_non_empty_data(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        assert framework_instance._extract_column_names(self.frame(adapter, "non_empty", _ONE_ROW))

    def test_public_extract_column_names_is_instance_free(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        """Public ``extract_column_names`` works on the class and through an instance."""
        empty_data = self.frame(adapter, "empty", _EMPTY)
        non_empty_data = self.frame(adapter, "non_empty", _ONE_ROW)
        framework_cls = type(framework_instance)

        columns = framework_cls.extract_column_names(non_empty_data)
        assert columns
        assert framework_cls.extract_column_names(empty_data)
        assert framework_instance.extract_column_names(non_empty_data) == columns


class _NonRootFeatureGroup:
    pass


def _data_creator_stub() -> type[DataCreatorFeatureGroup]:
    """A fresh stub, tracked so it never outlives its test in the class registry."""

    class _DataCreatorStub(DataCreatorFeatureGroup):
        @classmethod
        def calculate_feature(cls, data: Any, features: Any) -> Any:
            return {}

    return reclaim.track(_DataCreatorStub)


class AppendColumnsSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of _append_columns and its callers; hooks cover lazy plans, single columns and foreign data."""

    aspect = "append columns"
    needs_pinned_frame: ClassVar[bool] = False
    case_folds_names: ClassVar[bool] = False

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return True

    @pytest.fixture(autouse=True)
    # The "_a_" prefix sorts it first among autouse fixtures, so its mark is the outermost.
    def _a_reclaim_suite_built_classes(self) -> Iterator[None]:
        yield from reclaim.reclaimed_scope()

    def single_column(self, values: list[Any]) -> Any | None:
        """Return the framework's single-column type holding ``values``; None when it has no such branch."""
        return None

    def joined_frame(self, adapter: FrameworkAdapter) -> Any | None:
        """Override where the backend holds a lazy join plan; None means no separate plan."""
        return None

    def _joined(self, adapter: FrameworkAdapter) -> Any:
        joined = self.joined_frame(adapter)
        return self._appendable(adapter) if joined is None else joined

    def _single_column_or_fail(self, values: list[Any]) -> Any:
        column = self.single_column(values)
        if column is None:
            pytest.fail("implement single_column on this host or switch the test off with a Cause")
        return column

    def foreign_data(self, adapter: FrameworkAdapter) -> Any:
        """Data that is not this framework's frame type."""
        return {"a": [1, 2]}

    def _appendable(self, adapter: FrameworkAdapter) -> Any:
        return self.frame(adapter, "appendable", _APPENDABLE)

    def _single_column_keys(self, adapter: FrameworkAdapter, joined: Any) -> tuple[str, list[Any], list[str]]:
        rows = self.rows(adapter, joined)
        key = next(iter(rows[0]))
        keys = [row[key] for row in rows]
        return key, keys, [f"v{k}" for k in keys]

    def _pin(self, adapter: FrameworkAdapter, framework_instance: Any, joined: Any) -> None:
        rows = self.rows(adapter, joined)
        columns = {k: [r[k] for r in rows] for k in rows[0]}
        framework_instance.data = framework_instance._pinned_frame = framework_instance.transform(
            columns, list(columns)
        )

    def test_positional_append_needs_pinned_frame_opt_in(self, framework_instance: Any) -> None:
        assert framework_instance._positional_append_needs_pinned_frame() is self.needs_pinned_frame

    def test_positional_append_on_joined_data_keeps_key_alignment(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        joined = self._joined(adapter)
        rows = self.rows(adapter, joined)
        key = next(iter(rows[0]))
        keys = [row[key] for row in rows]
        values = [f"v{k}" for k in keys]

        if not self.needs_pinned_frame:
            result = framework_instance._append_columns(joined, {"appended": values})
            assert result is not None
            assert [(r[key], r["appended"]) for r in self.rows(adapter, result)] == list(zip(keys, values))
            return

        framework_instance.data = joined
        unpinned = {"appended": values}
        assert framework_instance._append_dict_to_frame(_NonRootFeatureGroup, unpinned) is unpinned

        self._pin(adapter, framework_instance, joined)
        out = framework_instance._append_dict_to_frame(_NonRootFeatureGroup, {"appended": values})

        assert [(r[key], r["appended"]) for r in self.rows(adapter, out)] == list(zip(keys, values))

    def test_single_column_on_unpinned_joined_frame(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        joined = self._joined(adapter)
        key, keys, values = self._single_column_keys(adapter, joined)
        column = self._single_column_or_fail(values)
        framework_instance.data = joined

        if self.needs_pinned_frame:
            with pytest.raises(ValueError, match="appended"):
                framework_instance.transform(column, ["appended"])
            return

        out = framework_instance.transform(column, ["appended"])
        assert [(r[key], r["appended"]) for r in self.rows(adapter, out)] == list(zip(keys, values))

    def test_single_column_on_pinned_frame_appends_aligned(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        joined = self._joined(adapter)
        key, keys, values = self._single_column_keys(adapter, joined)
        column = self._single_column_or_fail(values)
        self._pin(adapter, framework_instance, joined)

        out = framework_instance.transform(column, ["appended"])

        assert [(r[key], r["appended"]) for r in self.rows(adapter, out)] == list(zip(keys, values))

    def test_set_data_to_other_frame_clears_pin(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        joined = self.joined_frame(adapter)
        other = object() if joined is None else joined
        framework_instance.data = framework_instance._pinned_frame = self._appendable(adapter)

        framework_instance.set_data(other)

        assert framework_instance._pinned_frame is None

    def test_dict_append_on_unpinned_joined_frame(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        joined = self._joined(adapter)
        rows = self.row_count(adapter, joined)
        framework_instance.data = joined
        result = {"new_col": list(range(rows))}

        out = framework_instance._append_dict_to_frame(_NonRootFeatureGroup, result)

        if self.needs_pinned_frame:
            assert out is result
        else:
            assert [r["new_col"] for r in self.rows(adapter, out)] == result["new_col"]

    @pytest.mark.parametrize("group_name, appends", [("_NonRootFeatureGroup", True), ("_DataCreatorStub", False)])
    def test_dict_append_on_pinned_joined_frame(
        self, adapter: FrameworkAdapter, framework_instance: Any, group_name: str, appends: bool
    ) -> None:
        group = _NonRootFeatureGroup if appends else _data_creator_stub()
        joined = self._joined(adapter)
        rows = self.row_count(adapter, joined)
        framework_instance.data = joined
        framework_instance._pinned_frame = framework_instance.data
        values = list(range(rows))

        result = {"new_col": values}
        out = framework_instance._append_dict_to_frame(group, result)

        if not appends:
            assert out is result
            return
        assert out is not result
        assert [r["new_col"] for r in self.rows(adapter, out)] == values

    def test_append_result_is_appendable_again_with_aligned_rows(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        appendable = self._appendable(adapter)
        before = self.rows(adapter, appendable)
        xs = list(range(100, 100 + len(before)))
        ys = list(range(200, 200 + len(before)))

        first = framework_instance._append_columns(appendable, {"x": xs})
        assert first is not None
        second = framework_instance._append_columns(first, {"y": ys})

        assert second is not None
        after = self.rows(adapter, second)
        assert [(r["x"], r["y"]) for r in after] == list(zip(xs, ys))
        assert [{k: v for k, v in r.items() if k not in ("x", "y")} for r in after] == before

    def test_append_columns_keeps_existing_and_adds_new_in_order(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        appendable = self._appendable(adapter)
        before = self.rows(adapter, appendable)
        values = list(range(100, 100 + len(before)))

        result = framework_instance._append_columns(appendable, {"appended_col": values})

        assert result is not None
        after = self.rows(adapter, result)
        assert [row["appended_col"] for row in after] == values
        assert [{k: v for k, v in row.items() if k != "appended_col"} for row in after] == before
        for column, declared in _APPENDABLE.types.items():
            _assert_type(adapter, result, column, declared, appendable)

    def test_append_columns_returns_none_for_foreign_data_type(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        assert framework_instance._append_columns(self.foreign_data(adapter), {"appended_col": [1, 2]}) is None

    def test_append_columns_returns_none_on_length_mismatch(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        appendable = self._appendable(adapter)
        rows = self.row_count(adapter, appendable)

        assert framework_instance._append_columns(appendable, {"appended_col": list(range(rows + 1))}) is None

    def test_append_columns_case_differing_name(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        """Case-folding frameworks refuse a name differing only in case; the others append it next to the original."""
        appendable = self._appendable(adapter)
        values = list(range(self.row_count(adapter, appendable)))

        result = framework_instance._append_columns(appendable, {"A": values})

        if self.case_folds_names:
            assert result is None
            return
        assert result is not None
        assert self.column_values(adapter, result, "A") == values
        assert self.column_values(adapter, result, "a") == self.column_values(adapter, appendable, "a")
