"""Tests for discriminator-based join disambiguation in ExecutionPlan."""

from typing import Any, Dict, Tuple
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.user import Options


class TestMatchesDiscriminator:
    """Unit tests for ExecutionPlan._matches_discriminator."""

    def _make_graph_with_feature(self, options_dict: Dict[str, Any]) -> Tuple[MagicMock, UUID]:
        uuid = uuid4()
        graph = MagicMock()
        feature = MagicMock()
        feature.options = Options(group=options_dict)
        graph.nodes = {uuid: MagicMock(feature=feature)}
        return graph, uuid

    def test_matches_when_key_value_present(self) -> None:
        ep = ExecutionPlan()
        graph, uuid = self._make_graph_with_feature({"CsvReader": "customers.csv"})
        assert ep._matches_discriminator({"CsvReader": "customers.csv"}, graph, uuid) is True

    def test_no_match_when_key_missing(self) -> None:
        ep = ExecutionPlan()
        graph, uuid = self._make_graph_with_feature({"other_key": "value"})
        assert ep._matches_discriminator({"CsvReader": "customers.csv"}, graph, uuid) is False

    def test_no_match_when_value_differs(self) -> None:
        ep = ExecutionPlan()
        graph, uuid = self._make_graph_with_feature({"CsvReader": "orders.csv"})
        assert ep._matches_discriminator({"CsvReader": "customers.csv"}, graph, uuid) is False

    def test_no_match_on_empty_options(self) -> None:
        ep = ExecutionPlan()
        graph, uuid = self._make_graph_with_feature({})
        assert ep._matches_discriminator({"CsvReader": "customers.csv"}, graph, uuid) is False

    def test_matches_with_multiple_options(self) -> None:
        ep = ExecutionPlan()
        graph, uuid = self._make_graph_with_feature({"CsvReader": "customers.csv", "extra": "value"})
        assert ep._matches_discriminator({"CsvReader": "customers.csv"}, graph, uuid) is True
