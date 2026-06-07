"""
Shared test mixin for ComputeFramework._is_empty across compute frameworks.

The ``allow_empty_result`` feature lets a FeatureGroup declare that a zero-row result
is legitimate. The framework half of that contract is ``ComputeFramework._is_empty(data)``:
an overridable predicate that reports whether the framework-native data carries zero rows.
The base implementation returns ``False``; every framework that derives emptiness from row
count overrides it.

This mixin pins the observable contract per framework: ``_is_empty`` is ``True`` on a
zero-row (but column-bearing) frame and ``False`` on a frame with at least one row. It is
intentionally named without a ``Test`` prefix so pytest does not collect it standalone;
framework subclasses pick up the test methods by inheritance.

Mirror of ``DataTypeValidatorFrameworkTestMixin``: framework subclasses implement the
``framework_instance`` fixture and the ``empty_data`` / ``non_empty_data`` fixtures.
Connection-backed frameworks (DuckDB, SQLite, Spark) override the data fixtures to pull in
their ``connection`` / ``spark_session`` fixture, exactly as the DataTypeValidator mixin
consumers override ``validator_sample_data``.
"""

from typing import Any

import pytest


class EmptyResultFrameworkTestMixin:
    """Shared ``_is_empty`` tests for all compute frameworks."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        """Return a compute framework instance.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    def empty_data(self) -> Any:
        """Return framework-native data with a column but zero rows.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    def non_empty_data(self) -> Any:
        """Return framework-native data with at least one row.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    def test_is_empty_true_on_zero_row_data(self, framework_instance: Any, empty_data: Any) -> None:
        """``_is_empty`` must be True for a zero-row (but column-bearing) frame."""
        assert framework_instance._is_empty(empty_data) is True

    def test_is_empty_false_on_non_empty_data(self, framework_instance: Any, non_empty_data: Any) -> None:
        """``_is_empty`` must be False for a frame with at least one row."""
        assert framework_instance._is_empty(non_empty_data) is False
