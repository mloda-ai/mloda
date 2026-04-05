"""Tests for improved invariant error messages in ExecutionPlan.

These tests verify that the previously opaque "This should not happen" messages
in execution_plan.py now include actionable information: what invariant was
violated, the actual values, and a link to report the issue.
"""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph


class MockComputeFramework(ComputeFramework):
    pass


class TestCheckPointerDiscriminatorErrors:
    """Tests for check_pointer discriminator invariant messages."""

    def _make_execution_plan(self) -> ExecutionPlan:
        return ExecutionPlan()

    def test_right_discriminator_none_error_is_actionable(self) -> None:
        ep = self._make_execution_plan()
        graph = MagicMock(spec=Graph)

        link = MagicMock(spec=Link)
        link.left_discriminator = {"key": "value"}
        link.right_discriminator = None

        link_fw = (link, MockComputeFramework, MockComputeFramework)

        with pytest.raises(ValueError, match="Internal error.*right_discriminator is None"):
            ep.check_pointer({"key": "value"}, link_fw, graph, uuid4())

    def test_left_discriminator_none_error_is_actionable(self) -> None:
        ep = self._make_execution_plan()
        graph = MagicMock(spec=Graph)

        link = MagicMock(spec=Link)
        link.left_discriminator = None
        link.right_discriminator = {"key": "value"}

        link_fw = (link, MockComputeFramework, MockComputeFramework)

        with pytest.raises(ValueError, match="Internal error.*left_discriminator is None"):
            ep.check_pointer({"key": "value"}, link_fw, graph, uuid4())

    def test_discriminator_error_contains_actual_values(self) -> None:
        ep = self._make_execution_plan()
        graph = MagicMock(spec=Graph)

        link = MagicMock(spec=Link)
        link.left_discriminator = {"CsvReader": "file_a.csv"}
        link.right_discriminator = None

        link_fw = (link, MockComputeFramework, MockComputeFramework)

        with pytest.raises(ValueError, match="CsvReader.*file_a.csv"):
            ep.check_pointer({"CsvReader": "file_a.csv"}, link_fw, graph, uuid4())

    def test_discriminator_error_contains_report_url(self) -> None:
        ep = self._make_execution_plan()
        graph = MagicMock(spec=Graph)

        link = MagicMock(spec=Link)
        link.left_discriminator = {"key": "value"}
        link.right_discriminator = None

        link_fw = (link, MockComputeFramework, MockComputeFramework)

        with pytest.raises(ValueError, match="mloda-ai/mloda/issues"):
            ep.check_pointer({"key": "value"}, link_fw, graph, uuid4())

    def test_discriminator_error_contains_guidance(self) -> None:
        ep = self._make_execution_plan()
        graph = MagicMock(spec=Graph)

        link = MagicMock(spec=Link)
        link.left_discriminator = {"key": "value"}
        link.right_discriminator = None

        link_fw = (link, MockComputeFramework, MockComputeFramework)

        with pytest.raises(ValueError, match="both.*left_discriminator and right_discriminator"):
            ep.check_pointer({"key": "value"}, link_fw, graph, uuid4())
