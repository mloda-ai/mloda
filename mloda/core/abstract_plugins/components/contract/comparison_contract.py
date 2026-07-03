"""Cross-engine comparison contract (epic #518, Phase 0)."""

from dataclasses import dataclass
from enum import Enum


class SemanticDimension(Enum):
    """Semantic dimensions a column may need to satisfy for comparison."""

    ORDERED = "ordered"
    TEMPORAL = "temporal"
    NUMERIC = "numeric"


class TzPolicy(Enum):
    """Timezone policy a temporal column must follow."""

    ANY = "any"
    REQUIRE_AWARE = "require_aware"
    REQUIRE_NAIVE = "require_naive"


@dataclass(frozen=True)
class ColumnSemantics:
    """Observed semantics of a single column."""

    is_ordered: bool
    is_temporal: bool
    is_numeric: bool
    unit: str | None = None
    is_tz_aware: bool = False


@dataclass(frozen=True)
class ComparisonContract:
    """Contract that columns must satisfy to be compared or joined."""

    required: frozenset[SemanticDimension]
    unit: str | None = None
    tz_policy: TzPolicy = TzPolicy.ANY
    coerce: bool = False

    def validate(self, semantics: ColumnSemantics, column: str, side: str = "") -> None:
        label = f"{side} " if side else ""

        if SemanticDimension.ORDERED in self.required and not semantics.is_ordered:
            raise ValueError(f"Column '{column}' {label}must be ordered.")
        if SemanticDimension.TEMPORAL in self.required and not semantics.is_temporal:
            raise ValueError(f"Column '{column}' {label}must be temporal.")
        if SemanticDimension.NUMERIC in self.required and not semantics.is_numeric:
            raise ValueError(f"Column '{column}' {label}must be numeric.")

        if self.unit is not None:
            if semantics.unit is None:
                raise ValueError(f"Column '{column}' {label}must declare unit '{self.unit}'.")
            if semantics.unit != self.unit:
                raise ValueError(
                    f"Column '{column}' {label}has unit '{semantics.unit}' but contract requires '{self.unit}'."
                )

        if self.tz_policy is TzPolicy.REQUIRE_AWARE and not semantics.is_tz_aware:
            raise ValueError(f"Column '{column}' {label}must be timezone-aware.")
        if self.tz_policy is TzPolicy.REQUIRE_NAIVE and semantics.is_tz_aware:
            raise ValueError(f"Column '{column}' {label}must be timezone-naive.")

    def require_compatible(
        self, left: ColumnSemantics, right: ColumnSemantics, left_column: str, right_column: str
    ) -> None:
        if not (left.is_temporal and right.is_temporal):
            return

        if left.is_tz_aware != right.is_tz_aware:
            raise ValueError(f"Columns '{left_column}' and '{right_column}' have incompatible timezone awareness.")
