"""Shared time-reference helpers for time-based feature groups."""

from __future__ import annotations

from typing import Optional

from mloda.core.abstract_plugins.components.contract.comparison_contract import (
    ColumnSemantics,
    ComparisonContract,
    SemanticDimension,
)
from mloda.user import Options
from mloda.provider import DefaultOptionKeys, property_spec


class TimeReferenceMixin:
    """Provides the reference-time column resolver and time-unit mapping shared by time-based feature groups."""

    TIME_UNITS = {
        "second": "Seconds",
        "minute": "Minutes",
        "hour": "Hours",
        "day": "Days",
        "week": "Weeks",
        "month": "Months",
        "year": "Years",
    }

    REFERENCE_TIME_SPEC = property_spec(
        "Options key naming the reference-time column for time alignment; owned by TimeReferenceMixin, exposed by each concrete time-based feature group.",
        default=None,
    )

    @classmethod
    def get_reference_time_column(cls, options: Optional[Options] = None) -> str:
        """
        Get the reference time column name from options or use the default.

        Args:
            options: Optional Options object that may contain a custom reference time column name

        Returns:
            The reference time column name to use
        """
        reference_time_key = DefaultOptionKeys.reference_time
        if options and options.get(reference_time_key):
            reference_time = options.get(reference_time_key)
            if not isinstance(reference_time, str):
                raise ValueError(
                    f"Invalid reference_time option: {reference_time}. Must be string. Is: {type(reference_time)}."
                )
            return reference_time
        return DefaultOptionKeys.reference_time

    @classmethod
    def _validate_reference_time_column(cls, semantics: ColumnSemantics, reference_time_column: str) -> None:
        """Validate the reference-time column against the shared temporal + ordered contract."""
        ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL, SemanticDimension.ORDERED})).validate(
            semantics, reference_time_column
        )
