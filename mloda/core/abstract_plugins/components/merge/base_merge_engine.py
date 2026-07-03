from abc import ABC
from typing import Any, Optional, final

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics, ComparisonContract
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig, JoinType, Link


class BaseMergeEngine(ABC):
    """
    Abstract base class for merge operations.

    This class defines the structure for implementing various types of merge operations
    between two datasets, based on the specified join type. Subclasses are expected to
    implement the merge methods for specific join types as needed.
    """

    provides_column_semantics: bool = False

    def __init__(self, framework_connection: Optional[Any] = None) -> None:
        """
        Initialize the merge engine.

        Args:
            framework_connection: Optional connection object from the compute framework.
                                Some frameworks (e.g., DuckDB, Spark) need to share their
                                connection with merge engines for data consistency.
        """
        self.framework_connection = framework_connection

    def check_import(self) -> None:
        """
        Convenience method to check if the necessary imports are available. This is important for ensuring that not
        installed modules don't break the framework.

        This gets called in the final merge.

        Example:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        """
        pass

    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType inner is not yet implemented in {self.__class__.__name__}")

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType left is not yet implemented in {self.__class__.__name__}")

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType right is not yet implemented in {self.__class__.__name__}")

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType full outer is not yet implemented in {self.__class__.__name__}")

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType append is not yet implemented in {self.__class__.__name__}")

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType union is not yet implemented in {self.__class__.__name__}")

    def merge_asof(
        self,
        left_data: Any,
        right_data: Any,
        left_index: Index,
        right_index: Index,
        asof_config: AsOfJoinConfig,
    ) -> Any:
        raise ValueError(f"JoinType asof is not yet implemented in {self.__class__.__name__}")

    def validate_asof_time_columns(
        self, left_data: Any, right_data: Any, asof_config: AsOfJoinConfig
    ) -> tuple[Any, Any]:
        """Guard that as-of time columns are ordered (datetime / numeric / timedelta).

        Raises a clear ValueError naming the offending column instead of letting the
        backend surface a cryptic low-level dtype error. This is the opt-in coercion
        seam of issue #513: with asof_config.coerce_time_columns, a non-ordered side is
        replaced via _coerce_asof_time_column and the (possibly transformed) pair is
        returned.
        """
        sides = {"left": left_data, "right": right_data}
        cols = {"left": asof_config.left_time_column, "right": asof_config.right_time_column}
        sems: dict[str, ColumnSemantics] = {}
        for side in ("left", "right"):
            column = cols[side]
            sem = self._column_semantics(sides[side], column)
            if not sem.is_ordered:
                if not asof_config.coerce_time_columns:
                    raise ValueError(
                        f"As-of {side} time column '{column}' must be datetime, numeric, or timedelta, "
                        f"but is a non-ordered (e.g. string/object) dtype; cast it before joining. "
                        f"Set coerce_time_columns=True to opt in to ISO-8601 string coercion."
                    )
                sides[side] = self._coerce_asof_time_column(sides[side], column)
                sem = self._column_semantics(sides[side], column)
            sems[side] = sem
        ComparisonContract(required=frozenset()).require_compatible(
            sems["left"], sems["right"], cols["left"], cols["right"]
        )
        return sides["left"], sides["right"]

    def _column_semantics(self, data: Any, column: str) -> ColumnSemantics:
        """Observed semantics of a column, used by both as-of and equi-join validation.

        Opt-in hook (Option B, epic #518): the equi-join timezone/unit guard only calls
        this when the engine sets ``provides_column_semantics = True``. An engine that
        opts in promises to expose its framework-native ColumnSemantics; the base raises
        NotImplementedError so a forgotten override fails loudly. As-of engines override
        this unconditionally, so the as-of path always relies on it.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _column_semantics(data, column) "
            f"to support timezone/ordered validation for joins."
        )

    def _coerce_asof_time_column(self, data: Any, column: str) -> Any:
        """Coerce an ISO-8601 string as-of time column to a temporal/numeric dtype.

        Engines override this with framework-native coercion that fails hard on
        unparseable values. The base implementation raises so engines without
        coercion support fail loudly.
        """
        raise ValueError(
            f"{self.__class__.__name__} does not support coerce_time_columns; "
            f"cannot coerce as-of time column '{column}'."
        )

    def _asof_time_column_is_ordered(self, data: Any, column: str) -> bool:
        """Return True if `column` in `data` has an ordered (datetime/numeric/timedelta) dtype.

        Overridden per engine with framework-native dtype detection. Default raises so an
        asof-capable engine cannot silently skip the guard.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _asof_time_column_is_ordered for as-of joins."
        )

    def _validate_equi_join_time_columns(
        self, left_data: Any, right_data: Any, left_index: Index, right_index: Index
    ) -> None:
        """Guard that aligned equi-join key pairs are timezone/unit compatible.

        Reuses the as-of ComparisonContract machinery: require_compatible no-ops unless
        BOTH key columns are temporal, so string/integer/mismatched-type keys are
        unaffected (narrow policy). The guard is opt-in (Option B, epic #518): it is
        skipped entirely unless the engine sets ``provides_column_semantics = True``, so a
        time-agnostic engine author is never forced to implement ``_column_semantics``.
        When opted in, ``_column_semantics`` is required; the base raises
        NotImplementedError if an opted-in engine forgot to implement it, and the guard
        then only fires an error for temporal-vs-temporal key pairs.
        """
        if not self.provides_column_semantics:
            return
        left_cols = list(left_index.index)
        right_cols = list(right_index.index)
        if len(left_cols) != len(right_cols):
            return
        contract = ComparisonContract(required=frozenset())
        for left_col, right_col in zip(left_cols, right_cols):
            left_sem = self._column_semantics(left_data, left_col)
            right_sem = self._column_semantics(right_data, right_col)
            contract.require_compatible(left_sem, right_sem, left_col, right_col)

    @final
    def merge(self, left_data: Any, right_data: Any, link: Link) -> Any:
        self.check_import()

        jointype = link.jointype
        left_index = link.left_index
        right_index = link.right_index

        if jointype not in (JoinType.APPEND, JoinType.UNION):
            if len(left_index.index) != len(right_index.index):
                raise ValueError(
                    f"Left and right index lengths must match for join. "
                    f"Left has {len(left_index.index)} columns {left_index.index}, "
                    f"right has {len(right_index.index)} columns {right_index.index}."
                )

        if jointype in (JoinType.INNER, JoinType.LEFT, JoinType.RIGHT, JoinType.OUTER):
            self._validate_equi_join_time_columns(left_data, right_data, left_index, right_index)

        if jointype == JoinType.INNER:
            return self.merge_inner(left_data, right_data, left_index, right_index)
        elif jointype == JoinType.LEFT:
            return self.merge_left(left_data, right_data, left_index, right_index)
        elif jointype == JoinType.RIGHT:
            return self.merge_right(left_data, right_data, left_index, right_index)
        elif jointype == JoinType.OUTER:
            return self.merge_full_outer(left_data, right_data, left_index, right_index)
        elif jointype == JoinType.APPEND:
            return self.merge_append(left_data, right_data, left_index, right_index)
        elif jointype == JoinType.UNION:
            return self.merge_union(left_data, right_data, left_index, right_index)
        elif jointype == JoinType.ASOF:
            if link.asof_config is None:
                raise ValueError("ASOF join requires a Link carrying an asof_config.")
            return self.merge_asof(left_data, right_data, left_index, right_index, link.asof_config)
        else:
            raise ValueError(f"JoinType {jointype} is not yet implemented {self.__class__.__name__}")
