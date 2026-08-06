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


_DISCRIMINATOR_CASES = [
    pytest.param(
        {"key": "value"}, "right", "Internal error.*right_discriminator is None", id="right_side_none_is_actionable"
    ),
    pytest.param(
        {"key": "value"}, "left", "Internal error.*left_discriminator is None", id="left_side_none_is_actionable"
    ),
    pytest.param({"CsvReader": "file_a.csv"}, "right", "CsvReader.*file_a.csv", id="contains_actual_values"),
    pytest.param({"key": "value"}, "right", "mloda-ai/mloda/issues", id="contains_report_url"),
    pytest.param({"key": "value"}, "right", "both.*left_discriminator and right_discriminator", id="contains_guidance"),
]


class TestCheckPointerDiscriminatorErrors:
    """Tests for check_pointer discriminator invariant messages."""

    def _make_execution_plan(self) -> ExecutionPlan:
        return ExecutionPlan()

    @pytest.mark.parametrize(("discriminator", "none_side", "expected_match"), _DISCRIMINATOR_CASES)
    def test_discriminator_none_error(self, discriminator: dict[str, str], none_side: str, expected_match: str) -> None:
        ep = self._make_execution_plan()
        graph = MagicMock(spec=Graph)

        link = MagicMock(spec=Link)
        link.left_discriminator = None if none_side == "left" else discriminator
        link.right_discriminator = None if none_side == "right" else discriminator

        link_fw = (link, MockComputeFramework, MockComputeFramework)

        with pytest.raises(ValueError, match=expected_match):
            ep.check_pointer(discriminator, link_fw, graph, uuid4())
