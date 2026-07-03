"""Red-phase tests for the shared reference-time validator (epic #518, Phase 5).

These tests define the new ``TimeReferenceMixin._validate_reference_time_column``
classmethod that unifies the per-framework datetime checks onto the shared
``ComparisonContract`` (required = TEMPORAL + ORDERED).

They FAIL now because the method does not exist yet (AttributeError). The
attribute is reached via ``getattr`` so this test module stays clean under
``mypy --strict`` before the Green implementation adds the method.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.feature_group.experimental.time_reference_mixin import TimeReferenceMixin


def _validator() -> Any:
    """Return the shared validator classmethod (AttributeError until it exists)."""
    return getattr(TimeReferenceMixin, "_validate_reference_time_column")


class TestValidateReferenceTimeColumn:
    """Unit-level tests for the shared reference-time contract validator."""

    def test_validate_reference_time_column_exists(self) -> None:
        """The shared validator classmethod must exist on the mixin."""
        assert hasattr(TimeReferenceMixin, "_validate_reference_time_column")

    def test_accepts_tz_naive_temporal_column(self) -> None:
        """A tz-naive temporal + ordered column is accepted (no raise)."""
        semantics = ColumnSemantics(
            is_ordered=True,
            is_temporal=True,
            is_numeric=False,
            is_tz_aware=False,
        )
        _validator()(semantics, "reference_time")

    def test_accepts_tz_aware_temporal_column(self) -> None:
        """A tz-aware temporal + ordered column is accepted (no raise).

        Documents that the contract accepts both naive and aware temporal
        columns, unlike a strict tz policy.
        """
        semantics = ColumnSemantics(
            is_ordered=True,
            is_temporal=True,
            is_numeric=False,
            is_tz_aware=True,
        )
        _validator()(semantics, "reference_time")

    def test_rejects_non_temporal_column(self) -> None:
        """A non-temporal column raises ValueError naming the column."""
        semantics = ColumnSemantics(
            is_ordered=False,
            is_temporal=False,
            is_numeric=False,
            is_tz_aware=False,
        )
        with pytest.raises(ValueError, match="reference_time"):
            _validator()(semantics, "reference_time")
