"""
Shared test mixin pinning the schema-detection contract per compute framework.

The ``allow_empty_result`` policy now keys on SCHEMA PRESENCE, not row count. A result is
a valid (non-empty) result when it carries at least one column, even with zero rows. The
framework half of that contract is ``ComputeFramework.extract_column_names(data)`` (public
classmethod, instance-free): it returns the set of columns the framework sees on the native data.

This mixin pins the observable contract per framework:

- A schema-bearing zero-row frame yields a NON-EMPTY column set (state C: schema present,
  the guard must NOT treat this as empty).
- A frame with at least one row yields a NON-EMPTY column set.

The python_dict consumer sets ``empty_data_carries_schema = False`` because its native empty
(``[]``) carries no schema, so ``extract_column_names`` returns an empty set (state B).

It is intentionally named without a ``Test`` prefix so pytest does not collect it standalone;
framework subclasses pick up the test methods by inheritance.

Mirror of ``DataTypeValidatorFrameworkTestMixin``: framework subclasses implement the
``framework_instance`` fixture and the ``empty_data`` / ``non_empty_data`` fixtures.
Connection-backed frameworks (DuckDB, SQLite, Spark) override the data fixtures to pull in
their ``connection`` / ``spark_session`` fixture, exactly as the DataTypeValidator mixin
consumers override ``validator_sample_data``.
"""

from typing import Any

import pytest

from tests.test_plugins.compute_framework.test_tooling.policy_run_all_test_base import records_from_frame


class EmptyResultFrameworkTestMixin:
    """Shared schema-detection (``extract_column_names``) tests for all compute frameworks."""

    # Whether a zero-row frame in this framework still carries its schema (columns).
    # Schema-bearing frameworks (PyArrow, Pandas, Polars, DuckDB, SQLite, Spark, Iceberg): True.
    # python_dict (List[Dict]) drops the schema at [], so it overrides this to False.
    empty_data_carries_schema: bool = True

    @pytest.fixture
    def framework_instance(self) -> Any:
        """Return a compute framework instance.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    def empty_data(self) -> Any:
        """Return framework-native data with zero rows (column-bearing where the framework can).

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    def non_empty_data(self) -> Any:
        """Return framework-native data with at least one row.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    def test_extract_column_names_on_empty_data(self, framework_instance: Any, empty_data: Any) -> None:
        """A zero-row frame's schema presence drives the empty-result decision.

        Schema-bearing frameworks return a non-empty column set (state C). python_dict's
        native empty (``[]``) carries no schema, so the set is empty (state B).
        """
        columns = framework_instance._extract_column_names(empty_data)
        if self.empty_data_carries_schema:
            assert columns, "schema-bearing zero-row frame must expose its columns (state C)"
        else:
            assert columns == set(), "schema-less empty frame must expose no columns (state B)"

    def test_extract_column_names_on_non_empty_data(self, framework_instance: Any, non_empty_data: Any) -> None:
        """A frame with at least one row always exposes its columns."""
        assert framework_instance._extract_column_names(non_empty_data)

    def test_public_extract_column_names_is_instance_free(
        self, framework_instance: Any, empty_data: Any, non_empty_data: Any
    ) -> None:
        """Public ``extract_column_names`` works on the class and through an instance."""
        framework_cls = type(framework_instance)

        columns = framework_cls.extract_column_names(non_empty_data)
        assert columns

        empty_columns = framework_cls.extract_column_names(empty_data)
        if self.empty_data_carries_schema:
            assert empty_columns
        else:
            assert empty_columns == set()

        assert framework_instance.extract_column_names(non_empty_data) == columns


class _NonRootFeatureGroup:
    @classmethod
    def input_data(cls) -> None:
        return None


class AppendColumnsFrameworkTestMixin:
    """Shared ``_append_columns`` hook tests; reuses the ``framework_instance`` / ``non_empty_data`` fixtures."""

    # Frameworks whose held frame is a re-executed plan override this to True.
    needs_pinned_frame: bool = False

    @pytest.fixture
    def joined_data(self, appendable_data: Any) -> Any:
        """Override where the backend holds a lazy plan."""
        return appendable_data

    def test_positional_append_needs_pinned_frame_opt_in(self, framework_instance: Any) -> None:
        assert framework_instance._positional_append_needs_pinned_frame() is self.needs_pinned_frame

    def test_positional_append_on_joined_data_keeps_key_alignment(
        self, framework_instance: Any, joined_data: Any
    ) -> None:
        rows = records_from_frame(joined_data)
        key = next(iter(rows[0]))
        keys = [row[key] for row in rows]
        values = [f"v{k}" for k in keys]

        if not self.needs_pinned_frame:
            result = framework_instance._append_columns(joined_data, {"appended": values})
            assert result is not None
            assert [(r[key], r["appended"]) for r in records_from_frame(result)] == list(zip(keys, values))
            return

        framework_instance.data = joined_data
        unpinned = {"appended": values}
        assert framework_instance._append_dict_to_frame(_NonRootFeatureGroup, unpinned) is unpinned

        connection = getattr(joined_data, "connection", None)
        if connection is not None:
            framework_instance.set_framework_connection_object(connection)
        columns = {k: [r[k] for r in rows] for k in rows[0]}
        framework_instance.data = framework_instance._pinned_frame = framework_instance.transform(
            columns, list(columns)
        )

        out = framework_instance._append_dict_to_frame(_NonRootFeatureGroup, {"appended": values})

        assert [(r[key], r["appended"]) for r in records_from_frame(out)] == list(zip(keys, values))

    def test_set_data_to_other_frame_clears_pin(
        self, framework_instance: Any, appendable_data: Any, joined_data: Any
    ) -> None:
        other = joined_data if joined_data is not appendable_data else object()
        framework_instance.data = framework_instance._pinned_frame = appendable_data

        framework_instance.set_data(other)

        assert framework_instance._pinned_frame is None

    def test_dict_append_on_unpinned_joined_frame(self, framework_instance: Any, joined_data: Any) -> None:
        rows = len(records_from_frame(joined_data))
        framework_instance.data = joined_data
        result = {"new_col": list(range(rows))}

        out = framework_instance._append_dict_to_frame(_NonRootFeatureGroup, result)

        if self.needs_pinned_frame:
            assert out is result
        else:
            assert [r["new_col"] for r in records_from_frame(out)] == result["new_col"]

    def test_dict_append_on_pinned_joined_frame(self, framework_instance: Any, joined_data: Any) -> None:
        rows = len(records_from_frame(joined_data))
        framework_instance.data = joined_data
        framework_instance._pinned_frame = framework_instance.data
        values = list(range(rows))

        result = {"new_col": values}
        out = framework_instance._append_dict_to_frame(_NonRootFeatureGroup, result)

        assert out is not result
        assert [r["new_col"] for r in records_from_frame(out)] == values

    def test_append_result_is_appendable_again_with_aligned_rows(
        self, framework_instance: Any, appendable_data: Any
    ) -> None:
        before = records_from_frame(appendable_data)
        xs = list(range(100, 100 + len(before)))
        ys = list(range(200, 200 + len(before)))

        first = framework_instance._append_columns(appendable_data, {"x": xs})
        assert first is not None
        second = framework_instance._append_columns(first, {"y": ys})

        assert second is not None
        after = records_from_frame(second)
        assert [(r["x"], r["y"]) for r in after] == list(zip(xs, ys))
        assert [{k: v for k, v in r.items() if k not in ("x", "y")} for r in after] == before

    @pytest.fixture
    def appendable_data(self, non_empty_data: Any) -> Any:
        """Native frame the hook can append to; override when ``non_empty_data`` is not one."""
        return non_empty_data

    def test_append_columns_keeps_existing_and_adds_new_in_order(
        self, framework_instance: Any, appendable_data: Any
    ) -> None:
        before = records_from_frame(appendable_data)
        values = list(range(100, 100 + len(before)))

        result = framework_instance._append_columns(appendable_data, {"appended_col": values})

        assert result is not None
        after = records_from_frame(result)
        assert [row["appended_col"] for row in after] == values
        assert [{k: v for k, v in row.items() if k != "appended_col"} for row in after] == before

    @pytest.fixture
    def foreign_data(self) -> Any:
        """Data that is not this framework's frame type; default is a plain dict."""
        return {"a": [1, 2]}

    def test_append_columns_returns_none_for_foreign_data_type(
        self, framework_instance: Any, foreign_data: Any
    ) -> None:
        assert framework_instance._append_columns(foreign_data, {"appended_col": [1, 2]}) is None

    def test_append_columns_returns_none_on_length_mismatch(
        self, framework_instance: Any, appendable_data: Any
    ) -> None:
        rows = len(records_from_frame(appendable_data))

        assert framework_instance._append_columns(appendable_data, {"appended_col": list(range(rows + 1))}) is None


class AppendColumnsCaseFoldTestMixin:
    """Case-folding frameworks (DuckDB, SQLite, Spark) return None on a folded collision."""

    def test_append_columns_case_folded_collision_returns_none(
        self, framework_instance: Any, appendable_data: Any
    ) -> None:
        rows = len(records_from_frame(appendable_data))

        assert framework_instance._append_columns(appendable_data, {"A": list(range(rows))}) is None
