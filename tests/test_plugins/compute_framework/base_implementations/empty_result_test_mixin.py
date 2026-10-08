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


class AppendColumnsFrameworkTestMixin:
    """Shared ``_append_columns`` hook tests; reuses the ``framework_instance`` / ``non_empty_data`` fixtures."""

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
        self, framework_instance: Any, non_empty_data: Any
    ) -> None:
        rows = len(records_from_frame(non_empty_data))

        assert framework_instance._append_columns(non_empty_data, {"A": list(range(rows))}) is None
