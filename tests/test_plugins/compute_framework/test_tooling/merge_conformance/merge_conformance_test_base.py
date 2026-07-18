"""Cross-framework merge-engine conformance test base.

Runs the SAME canonical expected frames (see ``merge_conformance_scenarios``)
against one merge engine per subclass. Results are compared as an
order-independent, column-order-independent, null-normalized multiset of rows,
so each engine must agree on the exact contract regardless of internal
representation.

Mirrors the ``asof`` / ``multi_index`` strategy: this ABC holds the shared
logic and each compute framework subclasses it in its own test module,
supplying ``merge_engine_class`` / ``framework_type`` / ``get_connection``.
"""

import collections
import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import pytest

from mloda.provider import BaseMergeEngine
from mloda.user import Index, JoinType
from tests.test_plugins.compute_framework.test_tooling.merge_conformance.merge_conformance_scenarios import (
    JOIN_TYPE_NAMES,
    MERGE_CONFORMANCE_SCENARIOS,
)
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link
from tests.test_plugins.compute_framework.test_tooling.multi_index.test_data_converter import DataConverter


def _norm(value: Any) -> Any:
    """None stays None; float NaN collapses to None so nullable-int widening never matters.

    Tuned for the current numeric/string scenario data: boolean and temporal values would need
    ``_norm`` revisited before being added (Python treats ``True == 1``, so bools would collide).
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _multiset(rows: list[dict[str, Any]]) -> collections.Counter[frozenset[tuple[str, Any]]]:
    """Order- and column-order-independent, null-normalized bag of rows (1 == 1.0)."""
    return collections.Counter(frozenset((col, _norm(val)) for col, val in row.items()) for row in rows)


class MergeConformanceTestBase(ABC):
    """Base class for cross-framework merge conformance tests.

    Subclasses must implement:
    - merge_engine_class(): the MergeEngine class to test
    - framework_type(): the framework's native data type (pd.DataFrame, pa.Table, ...)
    - get_connection(): framework connection object, or None

    Every subclass runs the identical scenario x join-type matrix, so all engines
    are asserted against ONE canonical expected result.
    """

    def setup_method(self) -> None:
        """Initialize the test base with a data converter before each test method."""
        self.converter = DataConverter()

    # ==================== ABSTRACT METHODS (must implement) ====================

    @classmethod
    @abstractmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        """Return the merge engine class for this framework."""
        pass

    @classmethod
    @abstractmethod
    def framework_type(cls) -> type[Any]:
        """Return the framework's expected data type (e.g., pd.DataFrame, pa.Table)."""
        pass

    @abstractmethod
    def get_connection(self) -> Optional[Any]:
        """Return framework connection object, or None if not needed."""
        pass

    # ==================== TEST METHOD ====================

    @pytest.mark.parametrize("scenario_key", list(MERGE_CONFORMANCE_SCENARIOS))
    @pytest.mark.parametrize("join_name", JOIN_TYPE_NAMES)
    def test_merge_conformance(self, scenario_key: str, join_name: str) -> None:
        scenario = MERGE_CONFORMANCE_SCENARIOS[scenario_key]
        join_type = JoinType[join_name]
        framework_type = self.framework_type()

        connection = self.get_connection()

        left = self.converter.to_framework(scenario["left"], framework_type, connection)
        right = self.converter.to_framework(scenario["right"], framework_type, connection)
        link = make_merge_link(join_type, Index(scenario["left_index"]), Index(scenario["right_index"]))

        engine = self.merge_engine_class()(connection) if connection is not None else self.merge_engine_class()()
        result = engine.merge(left, right, link)
        actual = self.converter.from_framework(result, framework_type)

        expected = scenario["expected"][join_name]
        assert _multiset(actual) == _multiset(expected), (
            f"Merge conformance mismatch [{scenario_key} / {join_name}]:\n  expected={expected}\n  actual={actual}"
        )
