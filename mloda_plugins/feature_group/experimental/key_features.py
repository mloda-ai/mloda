"""Shared helper for feature groups that also read key columns next to their sources."""

from __future__ import annotations

from collections.abc import Iterable

from mloda.user import Feature, Options


def _forwarded(source: Feature) -> Options:
    forwarded = Options()
    forwarded.inherit_from(source.options)
    return forwarded


def with_key_features(sources: set[Feature], key_names: Iterable[str]) -> set[Feature]:
    """Key columns carry the group options the sources carry (minus in_features).

    They stay plain when the sources disagree.
    """
    ordered = sorted(sources, key=lambda f: str(f.name))
    forwarded = [_forwarded(f) for f in ordered]
    shared = bool(ordered) and all(o == forwarded[0] for o in forwarded) and bool(forwarded[0].group)
    taken = {str(f.name) for f in sources}
    result = set(sources)
    for name in key_names:
        if name not in taken:
            taken.add(name)
            result.add(Feature(name, _forwarded(ordered[0])) if shared else Feature(name))
    return result
