from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, TypeVar

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import (
    fold_identifier,
    pick_helper_column_name,
    quote_ident,
    require_exact_columns,
)
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import (
    OrderBy,
    WindowFrame,
    render_over_clause,
    validate_window,
)

RelT = TypeVar("RelT", bound="SqlBaseRelation")


class SqlBaseRelation(ABC):
    """Shared base for SQL relation wrappers (DuckDB, SQLite, and future backends).

    Provides collision-safe helper-column naming and the ``with_row_number`` /
    ``window`` window-function methods once, so every SQL backend inherits them.

    Contract for subclasses: a new ``*Relation`` MUST route any internally
    generated helper column through ``_pick_helper_column`` and guard
    user-supplied column names with ``_ensure_column_absent`` so collision-safe
    naming is automatic rather than a per-backend convention that is easy to drop.

    Subclasses implement two primitives:
    - ``columns``: the current column names.
    - ``_project_raw``: materialize ``SELECT {projection} FROM <self>`` as a new
      relation of the same type (the projection is a trusted SQL fragment).
    Subclasses MAY override the version-guard hooks (no-ops by default).
    """

    _PSEUDO_COLUMNS: ClassVar[frozenset[str]] = frozenset()

    @property
    @abstractmethod
    def columns(self) -> list[str]: ...

    @abstractmethod
    def _project_raw(self: RelT, projection: str) -> RelT:
        """Materialize ``SELECT {projection} FROM <self>`` as a new relation."""

    def _ensure_column_absent(self, name: str) -> None:
        """Raise ``ValueError`` if ``name`` collides with an existing column (ASCII case-insensitive)."""
        if fold_identifier(name) in {fold_identifier(c) for c in self.columns}:
            raise ValueError(f"Column {name!r} already exists in the relation")

    def _require_columns(self, *names: str) -> None:
        """Raise ``ValueError`` unless every name is an exact column (or a backend pseudo-column)."""
        require_exact_columns(self.columns, names, self._PSEUDO_COLUMNS)

    def _pick_helper_column(self, *also_taken: str) -> str:
        """Return a helper-column name free of every current column and ``also_taken`` (ASCII case-insensitive)."""
        return pick_helper_column_name(taken=set(self.columns) | set(also_taken))

    def _assert_window_supported(self) -> None:
        """Hook: raise if the backend cannot run window functions. No-op by default."""

    def _assert_nulls_supported(self, order_by: Sequence[str | OrderBy]) -> None:
        """Hook: raise if the backend cannot honor NULLS FIRST/LAST. No-op by default."""

    def with_row_number(
        self: RelT,
        alias: str,
        *,
        partition_by: Sequence[str] = (),
        order_by: Sequence[str | OrderBy] = (),
    ) -> RelT:
        """Append a ROW_NUMBER() window column named ``alias``.

        All identifiers in ``partition_by`` / ``order_by`` and the ``alias`` are quoted
        via ``quote_ident``. Raises ``ValueError`` if ``alias`` collides with an existing
        column (ASCII case-insensitive), or if a partition/order column is not an exact
        column name. With no partition_by/order_by, row-number assignment
        order is implementation-defined; pass order_by for a deterministic numbering.
        """
        self._ensure_column_absent(alias)
        self._require_columns(*partition_by, *(o if isinstance(o, str) else o.column for o in order_by))
        self._assert_window_supported()
        self._assert_nulls_supported(order_by)
        over_sql = render_over_clause(partition_by, order_by, None)
        return self._project_raw(f"*, ROW_NUMBER() OVER ({over_sql}) AS {quote_ident(alias)}")

    def window(
        self: RelT,
        func: str,
        alias: str,
        *,
        partition_by: Sequence[str] = (),
        order_by: Sequence[str | OrderBy] = (),
        frame: WindowFrame | None = None,
    ) -> RelT:
        """Append a window-function column ``alias`` computed by ``func`` over the OVER clause.

        ``func`` is a raw SQL fragment inlined verbatim; never pass user-controlled input.
        The ``alias`` and every identifier in ``partition_by`` / ``order_by`` are quoted via
        ``quote_ident``. Raises ``ValueError`` if ``alias`` collides with an existing column
        (ASCII case-insensitive), or if a partition/order column is not an exact column name.
        """
        self._ensure_column_absent(alias)
        self._require_columns(*partition_by, *(o if isinstance(o, str) else o.column for o in order_by))
        self._assert_window_supported()
        self._assert_nulls_supported(order_by)
        validate_window(order_by, frame)
        over_sql = render_over_clause(partition_by, order_by, frame)
        return self._project_raw(f"*, {func} OVER ({over_sql}) AS {quote_ident(alias)}")
