"""Shared time-reference helpers for time-based feature groups."""

from __future__ import annotations


from mloda.core.abstract_plugins.components.contract.comparison_contract import (
    ColumnSemantics,
    ComparisonContract,
    SemanticDimension,
)
from mloda.user import Options
from mloda.provider import DefaultOptionKeys, TimeRule, property_spec


class TimeReferenceMixin:
    """Provides the reference-time column resolver and time-unit mapping shared by time-based feature groups."""

    TIME_UNITS = TimeRule.UNITS
    TIME_RULE = "time_rule"

    TIME_RULE_SPEC = property_spec(
        "Time zone computation rule knobs as a mapping; applies to the feature that carries it, not its inputs.",
        default=None,
        context=False,
        forward_to_inputs=False,
        match_guard=TimeRule.from_option,
        expected=f"a mapping of the knobs {', '.join(TimeRule.KNOBS)} to allowed values",
    )

    REFERENCE_TIME_SPEC = property_spec(
        "Options key naming the reference-time column for time alignment; owned by TimeReferenceMixin, exposed by each concrete time-based feature group.",
        default=None,
    )

    @classmethod
    def get_reference_time_column(cls, options: Options | None = None) -> str:
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
    def get_time_rule(cls, options: Options | None = None) -> TimeRule:
        """The time rule carried by the options, the default rule when absent."""
        if options and cls.TIME_RULE in options.context:
            raise ValueError(
                f"The {cls.TIME_RULE} option must be a group option: only group options are part of a feature's identity."
            )
        return TimeRule.from_option(options.get(cls.TIME_RULE) if options else None)

    @classmethod
    def _validate_reference_time_column(cls, semantics: ColumnSemantics, reference_time_column: str) -> None:
        """Validate the reference-time column against the shared temporal + ordered contract."""
        ComparisonContract(required=frozenset({SemanticDimension.TEMPORAL, SemanticDimension.ORDERED})).validate(
            semantics, reference_time_column
        )
