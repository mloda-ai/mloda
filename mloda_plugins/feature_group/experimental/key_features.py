"""Shared helper for feature groups that also read key columns next to their sources."""

from __future__ import annotations

from collections.abc import Iterable

from mloda.user import Feature


def with_key_features(sources: set[Feature], key_names: Iterable[str]) -> set[Feature]:
    """Add one plain Feature per new key name, without any group options of the sources."""
    taken = {str(f.name) for f in sources}
    result = set(sources)
    for name in key_names:
        if name not in taken:
            taken.add(name)
            result.add(Feature(name))
    return result
