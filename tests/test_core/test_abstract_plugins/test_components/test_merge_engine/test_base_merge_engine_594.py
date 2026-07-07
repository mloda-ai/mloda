"""Lazy two-sided column-semantics seam (issue #594)."""

from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.provider import BaseMergeEngine
from mloda.user import Index


class _RecordingMergeEngine(BaseMergeEngine):
    provides_column_semantics = True

    def __init__(self, framework_connection: Any | None = None) -> None:
        super().__init__(framework_connection)
        self.semantics_calls: list[tuple[Any, str]] = []

    def _column_semantics(self, data: Any, column: str) -> ColumnSemantics:
        self.semantics_calls.append((data, column))
        return ColumnSemantics(is_ordered=False, is_temporal=False, is_numeric=False, unit=None, is_tz_aware=False)


class _SchemaSkipMergeEngine(_RecordingMergeEngine):
    def _schema_maybe_temporal(self, data: Any, column: str) -> bool:
        return False


class TestLazyTwoSidedColumnSemantics:
    def test_left_non_temporal_short_circuits_right(self) -> None:
        engine = _RecordingMergeEngine()
        left_index = Index(("lk",))
        right_index = Index(("rk",))

        engine._validate_equi_join_time_columns("left", "right", left_index, right_index)

        assert ("left", "lk") in engine.semantics_calls
        assert ("right", "rk") not in engine.semantics_calls
        assert len(engine.semantics_calls) == 1

    def test_schema_maybe_temporal_false_skips_both_sides(self) -> None:
        engine = _SchemaSkipMergeEngine()
        left_index = Index(("lk",))
        right_index = Index(("rk",))

        engine._validate_equi_join_time_columns("left", "right", left_index, right_index)

        assert engine.semantics_calls == []
