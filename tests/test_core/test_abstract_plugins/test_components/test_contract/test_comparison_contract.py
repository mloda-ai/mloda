"""Tests for the cross-engine comparison contract (epic #518, Phase 0).

These tests pin down the semantics of the comparison contract used to check
that columns from different compute engines can be safely compared or joined.
The implementation module does not exist yet, so these tests fail on import
until the Green phase creates it.
"""

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
