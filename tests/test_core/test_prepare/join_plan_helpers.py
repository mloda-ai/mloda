"""Shared feature and link-trekker helpers for the join plan tests."""

from typing import Any
from uuid import UUID

from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_links import LinkFrameworkTrekker, LinkTrekker
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


def join_tokens(plan: Any, link: Link) -> set[UUID]:
    return {step.uuid for step in plan if isinstance(step, JoinStep) and step.link.uuid == link.uuid}


def single_join_step(
    plan: ExecutionPlan,
    link_fw: LinkFrameworkTrekker,
    link_trekker: LinkTrekker,
    graph: Graph,
    pre_execution_plan: list[Any],
) -> JoinStep | None:
    join_steps = plan.run_link(link_fw, link_trekker, graph, pre_execution_plan)
    assert len(join_steps) <= 1
    return join_steps[0] if join_steps else None
