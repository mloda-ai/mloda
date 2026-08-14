"""Tests for discriminator-based join disambiguation in ExecutionPlan."""

from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.user import Options


class TestMatchesDiscriminator:
    """Unit tests for ExecutionPlan._matches_discriminator."""

    def _make_graph_with_feature(self, options_dict: dict[str, Any]) -> tuple[MagicMock, UUID]:
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

    def test_multi_key_discriminator_requires_every_key(self) -> None:
        """A discriminator with several keys must match all of them, not just one."""
        ep = ExecutionPlan()
        discriminator = {"reader": "file_a.csv", "region": "us"}

        graph, uuid = self._make_graph_with_feature({"reader": "file_a.csv", "region": "us"})
        assert ep._matches_discriminator(discriminator, graph, uuid) is True

        # Same region, different reader: the overlap on one key must not be enough.
        graph, uuid = self._make_graph_with_feature({"reader": "file_b.csv", "region": "us"})
        assert ep._matches_discriminator(discriminator, graph, uuid) is False

        # Same reader, different region: the mirror case.
        graph, uuid = self._make_graph_with_feature({"reader": "file_a.csv", "region": "eu"})
        assert ep._matches_discriminator(discriminator, graph, uuid) is False

    def test_multi_key_discriminator_needs_every_key_present(self) -> None:
        """Missing a discriminator key is a mismatch, even when the present ones agree."""
        ep = ExecutionPlan()
        graph, uuid = self._make_graph_with_feature({"reader": "file_a.csv"})
        assert ep._matches_discriminator({"reader": "file_a.csv", "region": "us"}, graph, uuid) is False

    def test_empty_discriminator_matches(self) -> None:
        """An empty discriminator constrains nothing, so every node satisfies it."""
        ep = ExecutionPlan()
        graph, uuid = self._make_graph_with_feature({"reader": "file_a.csv"})
        assert ep._matches_discriminator({}, graph, uuid) is True
