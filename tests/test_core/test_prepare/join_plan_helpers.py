"""Shared feature and link-trekker helpers for the join plan tests."""

from typing import Any
from uuid import UUID

from mloda.core.prepare.resolve_links import LinkTrekker
from mloda.provider import ComputeFramework
from mloda.user import Feature
from mloda.user import Index
from mloda.user import Link


def feature(
    name: str,
    cfw: type[ComputeFramework],
    index: Index | None = None,
    options: dict[str, Any] | None = None,
) -> Feature:
    built = Feature(name, index=index, options=options)
    built.compute_frameworks = {cfw}
    return built


def trek(
    link_trekker: LinkTrekker,
    link: Link,
    orientation: tuple[type[ComputeFramework], type[ComputeFramework]],
    uuid: UUID,
) -> None:
    """Production shares one set object between data and data_ordered, and invert_link relies on that."""
    key = (link, orientation[0], orientation[1])
    trekked = link_trekker.data.get(key)
    if trekked is None:
        trekked = set()
        link_trekker.data[key] = trekked
        link_trekker.data_ordered[key] = trekked
    trekked.add(uuid)
