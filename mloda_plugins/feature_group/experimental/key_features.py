"""Shared helper for feature groups that also read key columns next to their sources."""

from __future__ import annotations

from collections.abc import Iterable

from mloda.user import Feature, Options


def with_key_features(sources: set[Feature], key_names: Iterable[str]) -> set[Feature]:
    """Add one Feature per new key name, carrying the group options of the first source."""
    taken = {str(f.name) for f in sources}
    group = next((dict(f.options.group) for f in sorted(sources, key=lambda f: str(f.name)) if f.options.group), None)
    result = set(sources)
    for name in key_names:
        if name not in taken:
            taken.add(name)
            result.add(Feature(name, Options(group=dict(group))) if group else Feature(name))
    return result
