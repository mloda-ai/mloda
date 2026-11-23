"""Tests for the cross-engine comparison contract (epic #518, Phase 0).

These tests pin down the semantics of the comparison contract used to check
that columns from different compute engines can be safely compared or joined.
The implementation module does not exist yet, so these tests fail on import
until the Green phase creates it.
"""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.contract.comparison_contract import (
    ColumnSemantics,
    ComparisonContract,
    SemanticDimension,
    TzPolicy,
)


class TestEnums:
    """The two enums must exist with the exact member values from the contract."""

    def test_semantic_dimension_values(self) -> None:
        assert SemanticDimension.ORDERED.value == "ordered"
        assert SemanticDimension.TEMPORAL.value == "temporal"
        assert SemanticDimension.NUMERIC.value == "numeric"

    def test_tz_policy_values(self) -> None:
        assert TzPolicy.ANY.value == "any"
        assert TzPolicy.REQUIRE_AWARE.value == "require_aware"
        assert TzPolicy.REQUIRE_NAIVE.value == "require_naive"


class TestDataclassDefaults:
    """ColumnSemantics and ComparisonContract are frozen dataclasses with defaults."""

    def test_column_semantics_defaults(self) -> None:
        semantics = ColumnSemantics(is_ordered=True, is_temporal=False, is_numeric=True)
        assert semantics.unit is None
        assert semantics.is_tz_aware is False

    def test_column_semantics_is_frozen(self) -> None:
        semantics = ColumnSemantics(is_ordered=True, is_temporal=False, is_numeric=True)
        with pytest.raises((AttributeError, TypeError)):
            semantics.is_ordered = False  # type: ignore[misc]

    def test_comparison_contract_defaults(self) -> None:
        contract = ComparisonContract(required=frozenset())
        assert contract.unit is None
        assert contract.tz_policy is TzPolicy.ANY
        assert contract.coerce is False

    def test_comparison_contract_is_frozen(self) -> None:
        contract = ComparisonContract(required=frozenset())
        with pytest.raises((AttributeError, TypeError)):
            contract.coerce = True  # type: ignore[misc]


class TestValidateDimensions:
    """validate() enforces the required semantic dimensions."""

    def test_ordered_required_but_missing_raises(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.ORDERED}))
        semantics = ColumnSemantics(is_ordered=False, is_temporal=False, is_numeric=False)
        with pytest.raises(ValueError) as exc_info:
            contract.validate(semantics, "event_id")
        message = str(exc_info.value)
        assert "event_id" in message
        assert "ordered" in message

    def test_temporal_required_but_missing_raises(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        semantics = ColumnSemantics(is_ordered=False, is_temporal=False, is_numeric=False)
        with pytest.raises(ValueError) as exc_info:
            contract.validate(semantics, "ts")
        message = str(exc_info.value)
        assert "ts" in message
        assert "temporal" in message

    def test_numeric_required_but_missing_raises(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.NUMERIC}))
        semantics = ColumnSemantics(is_ordered=False, is_temporal=False, is_numeric=False)
        with pytest.raises(ValueError) as exc_info:
            contract.validate(semantics, "amount")
        message = str(exc_info.value)
        assert "amount" in message
        assert "numeric" in message

    def test_all_dimensions_satisfied_returns_none(self) -> None:
        contract = ComparisonContract(
            required=frozenset({SemanticDimension.ORDERED, SemanticDimension.TEMPORAL, SemanticDimension.NUMERIC})
        )
        semantics = ColumnSemantics(is_ordered=True, is_temporal=True, is_numeric=True)
        contract.validate(semantics, "ts")  # must not raise

    def test_empty_required_returns_none(self) -> None:
        contract = ComparisonContract(required=frozenset())
        semantics = ColumnSemantics(is_ordered=False, is_temporal=False, is_numeric=False)
        contract.validate(semantics, "anything")  # must not raise


class TestValidateUnit:
    """validate() enforces unit constraints when contract.unit is set."""

    def test_unit_required_but_column_unit_none_raises(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}), unit="s")
        semantics = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit=None)
        with pytest.raises(ValueError) as exc_info:
            contract.validate(semantics, "ts")
        assert "unit" in str(exc_info.value)

    def test_unit_mismatch_raises_with_both_units(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}), unit="s")
        semantics = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit="ms")
        with pytest.raises(ValueError) as exc_info:
            contract.validate(semantics, "ts")
        message = str(exc_info.value)
        assert "s" in message
        assert "ms" in message
        assert "ts" in message

    def test_unit_match_returns_none(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}), unit="s")
        semantics = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit="s")
        contract.validate(semantics, "ts")  # must not raise

    def test_no_contract_unit_ignores_column_unit(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        semantics = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit="ns")
        contract.validate(semantics, "ts")  # must not raise


class TestValidateTzPolicy:
    """validate() enforces the timezone policy."""

    def test_require_aware_but_naive_raises(self) -> None:
        contract = ComparisonContract(
            required=frozenset({SemanticDimension.TEMPORAL}), tz_policy=TzPolicy.REQUIRE_AWARE
        )
        semantics = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=False)
        with pytest.raises(ValueError) as exc_info:
            contract.validate(semantics, "ts")
        message = str(exc_info.value).lower()
        assert "timezone" in message or "tz" in message

    def test_require_naive_but_aware_raises(self) -> None:
        contract = ComparisonContract(
            required=frozenset({SemanticDimension.TEMPORAL}), tz_policy=TzPolicy.REQUIRE_NAIVE
        )
        semantics = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=True)
        with pytest.raises(ValueError) as exc_info:
            contract.validate(semantics, "ts")
        message = str(exc_info.value).lower()
        assert "timezone" in message or "tz" in message

    def test_require_aware_and_aware_returns_none(self) -> None:
        contract = ComparisonContract(
            required=frozenset({SemanticDimension.TEMPORAL}), tz_policy=TzPolicy.REQUIRE_AWARE
        )
        semantics = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=True)
        contract.validate(semantics, "ts")  # must not raise

    def test_require_naive_and_naive_returns_none(self) -> None:
        contract = ComparisonContract(
            required=frozenset({SemanticDimension.TEMPORAL}), tz_policy=TzPolicy.REQUIRE_NAIVE
        )
        semantics = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=False)
        contract.validate(semantics, "ts")  # must not raise

    def test_tz_policy_any_ignores_aware(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        aware = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=True)
        naive = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=False)
        contract.validate(aware, "ts")  # must not raise
        contract.validate(naive, "ts")  # must not raise


class TestValidateSide:
    """The optional side label appears in messages only when provided."""

    def test_side_present_in_message_when_provided(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.ORDERED}))
        semantics = ColumnSemantics(is_ordered=False, is_temporal=False, is_numeric=False)
        with pytest.raises(ValueError) as exc_info:
            contract.validate(semantics, "event_id", side="left")
        assert "left" in str(exc_info.value)

    def test_side_omitted_cleanly_when_empty(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.ORDERED}))
        semantics = ColumnSemantics(is_ordered=False, is_temporal=False, is_numeric=False)
        with pytest.raises(ValueError) as exc_info:
            contract.validate(semantics, "event_id", side="")
        message = str(exc_info.value)
        assert "event_id" in message
        # An empty side must not leak placeholder artifacts into the message.
        assert "()" not in message
        assert "[]" not in message


class TestRequireCompatibleTimezone:
    """require_compatible() uses a strict naive-vs-aware timezone model."""

    def test_aware_vs_naive_mix_raises(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        left = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=True)
        right = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=False)
        with pytest.raises(ValueError) as exc_info:
            contract.require_compatible(left, right, "left_ts", "right_ts")
        message = str(exc_info.value)
        assert "left_ts" in message
        assert "right_ts" in message
        assert "timezone" in message.lower() or "tz" in message.lower()

    def test_naive_vs_aware_mix_raises(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        left = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=False)
        right = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=True)
        with pytest.raises(ValueError):
            contract.require_compatible(left, right, "left_ts", "right_ts")

    def test_both_aware_returns_none(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        left = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=True)
        right = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=True)
        contract.require_compatible(left, right, "left_ts", "right_ts")  # must not raise

    def test_both_naive_returns_none(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        left = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=False)
        right = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, is_tz_aware=False)
        contract.require_compatible(left, right, "left_ts", "right_ts")  # must not raise


class TestRequireCompatibleUnit:
    """require_compatible() no longer enforces cross-side units.

    Cross-side unit enforcement was dropped: it caused false positives on
    ns-vs-us joins and contradicts the deferral. require_compatible only
    guards naive-vs-aware timezone mixing now; differing units must NOT raise.
    """

    def test_differing_units_do_not_raise(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        left = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit="s")
        right = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit="ms")
        contract.require_compatible(left, right, "left_ts", "right_ts")  # must not raise

    def test_matching_units_return_none(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        left = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit="s")
        right = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit="s")
        contract.require_compatible(left, right, "left_ts", "right_ts")  # must not raise

    def test_left_unit_none_skips_unit_check(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        left = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit=None)
        right = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit="ms")
        contract.require_compatible(left, right, "left_ts", "right_ts")  # must not raise

    def test_both_units_none_skips_unit_check(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        left = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit=None)
        right = ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, unit=None)
        contract.require_compatible(left, right, "left_ts", "right_ts")  # must not raise


class TestRequireCompatibleNonTemporal:
    """require_compatible() imposes no tz/unit checks on non-temporal columns."""

    def test_numeric_ordered_keys_return_none(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.ORDERED, SemanticDimension.NUMERIC}))
        left = ColumnSemantics(is_ordered=True, is_temporal=False, is_numeric=True)
        right = ColumnSemantics(is_ordered=True, is_temporal=False, is_numeric=True)
        contract.require_compatible(left, right, "left_id", "right_id")  # must not raise


def _temporal_semantics(**metadata: Any) -> ColumnSemantics:
    return ColumnSemantics(is_ordered=True, is_temporal=True, is_numeric=False, **metadata)


class TestColumnTimezoneIdentity:
    def test_default_zone_is_unknown(self) -> None:
        sem = _temporal_semantics()
        assert getattr(sem, "tz") is None
        assert sem.is_tz_aware is False

    @pytest.mark.parametrize("aware", [False, True])
    def test_legacy_positional_awareness_is_preserved(self, aware: bool) -> None:
        sem = ColumnSemantics(True, True, False, "us", aware)
        assert sem.is_tz_aware is aware
        assert sem.unit == "us"
        assert getattr(sem, "tz") is None

    def test_legacy_keyword_can_report_aware_with_unknown_zone(self) -> None:
        sem = _temporal_semantics(is_tz_aware=True)
        assert sem.is_tz_aware is True
        assert getattr(sem, "tz") is None

    @pytest.mark.parametrize("zone", ["UTC", "Europe/Berlin", "Asia/Kolkata"])
    def test_known_zone_implies_awareness(self, zone: str) -> None:
        sem = _temporal_semantics(tz=zone)
        assert getattr(sem, "tz") == zone
        assert sem.is_tz_aware is True

    def test_zone_is_immutable(self) -> None:
        sem = _temporal_semantics(tz="Europe/Berlin")
        with pytest.raises(FrozenInstanceError):
            setattr(sem, "tz", "UTC")

    @pytest.mark.parametrize(
        ("left_zone", "right_zone"),
        [("Europe/Berlin", "Asia/Kolkata"), ("UTC", None), (None, None)],
    )
    def test_aware_zones_remain_comparable(self, left_zone: str | None, right_zone: str | None) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        left = _temporal_semantics(tz=left_zone, is_tz_aware=True, unit="ns")
        right = _temporal_semantics(tz=right_zone, is_tz_aware=True, unit="us")
        contract.require_compatible(left, right, "left_ts", "right_ts")

    @pytest.mark.parametrize("zone", ["Europe/Berlin", None])
    @pytest.mark.parametrize("aware_on_left", [False, True])
    def test_aware_and_naive_remain_incompatible(self, zone: str | None, aware_on_left: bool) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        aware = _temporal_semantics(tz=zone, is_tz_aware=True)
        naive = _temporal_semantics()
        left, right = (aware, naive) if aware_on_left else (naive, aware)
        with pytest.raises(ValueError, match="incompatible timezone awareness"):
            contract.require_compatible(left, right, "left_ts", "right_ts")

    @pytest.mark.parametrize("zone", ["Europe/Berlin", None])
    def test_policy_uses_awareness_even_when_zone_is_unknown(self, zone: str | None) -> None:
        sem = _temporal_semantics(tz=zone, is_tz_aware=True)
        ComparisonContract(frozenset(), tz_policy=TzPolicy.REQUIRE_AWARE).validate(sem, "ts")
        with pytest.raises(ValueError, match="timezone-naive"):
            ComparisonContract(frozenset(), tz_policy=TzPolicy.REQUIRE_NAIVE).validate(sem, "ts")
