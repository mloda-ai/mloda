"""Tests for the cross-engine comparison contract (epic #518, Phase 0).

These tests pin down the semantics of the comparison contract used to check
that columns from different compute engines can be safely compared or joined.
The implementation module does not exist yet, so these tests fail on import
until the Green phase creates it.
"""

from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo

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


def _temporal(**kwargs: Any) -> ColumnSemantics:
    return ColumnSemantics(is_ordered=False, is_temporal=True, is_numeric=False, **kwargs)


class TestColumnSemanticsZone:
    """ColumnSemantics.tz carries the zone; a given zone implies awareness."""

    def test_tz_defaults_to_none(self) -> None:
        assert _temporal().tz is None

    def test_tz_given_implies_aware(self) -> None:
        assert _temporal(tz="America/New_York").is_tz_aware is True

    def test_tz_given_overrides_explicit_naive(self) -> None:
        assert _temporal(is_tz_aware=False, tz="X").is_tz_aware is True

    def test_aware_with_unknown_zone_is_expressible(self) -> None:
        semantics = _temporal(is_tz_aware=True, tz=None)
        assert semantics.is_tz_aware is True
        assert semantics.tz is None

    def test_different_zones_stay_compatible(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        contract.require_compatible(_temporal(tz="UTC"), _temporal(tz="Asia/Kolkata"), "a", "b")

    def test_known_and_unknown_zone_stay_compatible(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        contract.require_compatible(_temporal(tz="UTC"), _temporal(is_tz_aware=True), "a", "b")

    def test_zoned_vs_naive_still_rejected(self) -> None:
        contract = ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL}))
        with pytest.raises(ValueError):
            contract.require_compatible(_temporal(tz="UTC"), _temporal(), "a", "b")


class _OddTz(tzinfo):
    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return None

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, dt: datetime | None) -> str | None:
        return None

    def __str__(self) -> str:
        return "odd-zone"


class TestZoneName:
    """zone_name spells a tzinfo as an IANA key, UTC, a fixed offset or its str."""

    @pytest.mark.parametrize(
        "tz, expected",
        [
            (ZoneInfo("America/New_York"), "America/New_York"),
            (ZoneInfo("Europe/Berlin"), "Europe/Berlin"),
            (UTC, "UTC"),
            (timezone(timedelta(hours=5, minutes=30)), "+05:30"),
            (timezone(timedelta(hours=-8)), "-08:00"),
            (timezone(-timedelta(hours=5, minutes=30, seconds=30)), "-05:30"),
            (_OddTz(), "odd-zone"),
        ],
    )
    def test_zone_name(self, tz: tzinfo, expected: str) -> None:
        from mloda.core.abstract_plugins.components.contract.comparison_contract import zone_name

        assert zone_name(tz) == expected

    def test_dateutil_tzutc_is_utc(self) -> None:
        tz = pytest.importorskip("dateutil.tz")
        from mloda.core.abstract_plugins.components.contract.comparison_contract import zone_name

        assert zone_name(tz.tzutc()) == "UTC"
