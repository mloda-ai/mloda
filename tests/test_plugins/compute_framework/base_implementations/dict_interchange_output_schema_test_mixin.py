"""
Shared test mixin for ComputeFramework._output_schema on the dict interchange shape.

A dict result (column name to values) is read by the module-level ``_dict_output_schema``
helper before any framework-specific extraction runs, so this behavior is identical on every
compute framework. Framework subclasses only need to provide a ``framework_instance`` fixture.
"""

from typing import Any

import pytest


class DictInterchangeOutputSchemaTestMixin:
    """Shared _output_schema tests for the dict interchange shape across all compute frameworks.

    The mixin is intentionally named without a ``Test`` prefix so pytest does not collect it
    standalone. Framework subclasses pick up the test methods by inheritance.
    """

    @pytest.fixture
    def framework_instance(self) -> Any:
        """Return a compute framework instance.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    def test_reads_dict_sorted_with_python_type_names(self, framework_instance: Any) -> None:
        assert framework_instance._output_schema({"b": ["x"], "a": [1], "c": [1.5]}) == (
            ("a", "int"),
            ("b", "str"),
            ("c", "float"),
        )
