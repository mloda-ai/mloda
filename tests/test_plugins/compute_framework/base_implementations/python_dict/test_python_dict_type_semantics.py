"""Failing tests for the python_dict column-semantics introspector (epic #518, Phase 1).

Defines the not-yet-implemented module-level function ``column_semantics(rows, column)``
in ``mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_type_semantics``.
Because python_dict carries no schema, the function inspects the first non-null value.
Expected to fail at import time (ModuleNotFoundError) until Green implements it.
"""

from datetime import datetime, timezone
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_type_semantics import (
    column_semantics,
)


class TestPythonDictColumnSemantics:
    def data(self) -> dict[str, list[Any]]:
        return {
            "ts_naive": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
            "ts_aware": [
                datetime(2021, 1, 1, tzinfo=timezone.utc),
                datetime(2021, 1, 2, tzinfo=timezone.utc),
            ],
            "num": [1, 2],
            "s": ["a", "b"],
        }

    def test_ts_naive(self) -> None:
        sem = column_semantics(self.data(), "ts_naive")
        assert isinstance(sem, ColumnSemantics)
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_ts_aware(self) -> None:
        sem = column_semantics(self.data(), "ts_aware")
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is True
        assert sem.unit is None

    def test_num(self) -> None:
        sem = column_semantics(self.data(), "num")
        assert sem.is_ordered is True
        assert sem.is_temporal is False
        assert sem.is_numeric is True
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_s(self) -> None:
        sem = column_semantics(self.data(), "s")
        assert sem.is_ordered is False
        assert sem.is_temporal is False
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None
