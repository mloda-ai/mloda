"""The parse of a feature name as immutable facts (issue #770)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedFeatureName:
    """A frozen record of what parsing a feature name found, mirroring what ``re`` reports.

    A named group appears in BOTH ``named_captures`` and ``positional_captures``, exactly as
    ``re.Match.groupdict()`` and ``re.Match.groups()`` do; a non-participating optional group is
    ``None`` in both views. ``operation_part`` is the raw suffix text after the last separator, never
    a fabricated operation token.
    """

    matched: bool
    source_feature: str | None = None
    operation_part: str | None = None
    named_captures: Mapping[str, str | None] = field(default_factory=dict)
    positional_captures: tuple[str | None, ...] = ()

    @classmethod
    def no_match(cls) -> "ParsedFeatureName":
        """The miss case: no pattern matched, so there are no facts."""
        return cls(matched=False)
