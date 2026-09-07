"""
Shared test mixins for ComputeFramework._output_schema edge cases that recur identically
across frameworks: an empty schema yielding None, and duplicate column names collapsing to
one entry. Framework subclasses only need to provide a ``framework_instance`` fixture plus
whichever of ``empty_schema_data`` / ``duplicate_column_data`` their mixin requires.
"""

from typing import Any

import pytest


class EmptySchemaOutputSchemaTestMixin:
    """Shared test: a schema with no columns yields None.

    The mixin is intentionally named without a ``Test`` prefix so pytest does not collect it
    standalone. Framework subclasses pick up the test method by inheritance.
    """

    @pytest.fixture
    def framework_instance(self) -> Any:
        """Return a compute framework instance.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    def empty_schema_data(self) -> Any:
        """Return framework-specific native data with no columns.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    def test_empty_schema_yields_none(self, framework_instance: Any, empty_schema_data: Any) -> None:
        assert framework_instance._output_schema(empty_schema_data) is None


class DuplicateColumnOutputSchemaTestMixin:
    """Shared test: duplicate column names (e.g. an un-aliased join) collapse to one entry.

    The mixin is intentionally named without a ``Test`` prefix so pytest does not collect it
    standalone. Framework subclasses pick up the test method by inheritance.
    """

    @pytest.fixture
    def framework_instance(self) -> Any:
        """Return a compute framework instance.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    def duplicate_column_data(self) -> Any:
        """Return framework-specific native data with a duplicated "a" column.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    def test_duplicate_column_names_collapse_to_one_entry(
        self, framework_instance: Any, duplicate_column_data: Any
    ) -> None:
        result = framework_instance._output_schema(duplicate_column_data)
        assert result is not None
        a_entries = [pair for pair in result if pair[0] == "a"]
        assert len(a_entries) == 1
