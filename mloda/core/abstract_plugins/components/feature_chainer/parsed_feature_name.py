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


@dataclass(frozen=True)
class NameResolution:
    """One name resolution: parse facts, ownership, name bindings and the raw source split (empties kept)."""

    parsed: ParsedFeatureName
    owned: bool = False
    bindings: Mapping[str, str] = field(default_factory=dict)
    sources: tuple[str, ...] = ()

    def value_for(self, key: str) -> str | None:
        """The name-carried value for ``key``: its own capture when named, the first capture when positional."""
        if self.parsed.named_captures:
            return self.parsed.named_captures.get(key)
        if self.parsed.positional_captures:
            return self.parsed.positional_captures[0]
        return None

    @classmethod
    def miss(cls) -> "NameResolution":
        """The name owns nothing."""
        return cls(parsed=ParsedFeatureName.no_match())
