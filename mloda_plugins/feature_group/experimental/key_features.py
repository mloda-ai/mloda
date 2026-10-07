"""Shared helper for feature groups that also read key columns next to their sources."""

from __future__ import annotations

from collections.abc import Iterable

from mloda.user import Feature, Options
from mloda.provider import DefaultOptionKeys


def _forwarded(source: Feature) -> Options:
    forwarded = Options()
    forwarded.inherit_from(source.options)
    return forwarded


def with_key_features(sources: set[Feature], key_names: Iterable[str]) -> set[Feature]:
    """Add a key Feature per new name, carrying the options chained sources forward (none otherwise)."""
    chained = [_forwarded(f) for f in sources if DefaultOptionKeys.in_features in f.options]
    chained_sources = [f for f in sources if DefaultOptionKeys.in_features in f.options]
    shared = bool(chained) and all(o == chained[0] for o in chained)
    taken = {str(f.name) for f in sources}
    result = set(sources)
    for name in key_names:
        if name not in taken:
            taken.add(name)
            result.add(Feature(name, _forwarded(chained_sources[0])) if shared else Feature(name))
    return result
