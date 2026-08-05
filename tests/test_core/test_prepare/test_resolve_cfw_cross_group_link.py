"""One orientation per link: groups sharing a trekker agree on a single compute framework."""

from typing import Any
from uuid import UUID

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_compute_frameworks import ResolveComputeFrameworks
from mloda.core.prepare.resolve_links import LinkTrekker
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class CrossGroupLeftFG(FeatureGroup):
    pass


class CrossGroupRightFG(FeatureGroup):
    pass


class CrossGroupChildAFG(FeatureGroup):
    pass


class CrossGroupChildBFG(FeatureGroup):
    pass


def _make_trekker(trekked_uuids: set[UUID]) -> tuple[LinkTrekker, Any]:
    link = Link.inner(JoinSpec(CrossGroupLeftFG, "idx"), JoinSpec(CrossGroupRightFG, "idx"))
    trekker = (link, PandasDataFrame, PyArrowTable)

    link_trekker = LinkTrekker()
    link_trekker.data[trekker] = set(trekked_uuids)
    link_trekker.data_ordered[trekker] = set(trekked_uuids)
    return link_trekker, trekker


def _orientations_present(link_trekker: LinkTrekker, link: Link) -> list[Any]:
    return [key for key in link_trekker.data_ordered if key[0] == link]


def _resolve_cross_group_scenario() -> tuple[Feature, Feature, LinkTrekker, Any]:
    """Group A could take either framework, group B only the right one."""
    feature_a = Feature("cross_group_feature_a")
    feature_b = Feature("cross_group_feature_b")
    feature_a.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature_b.compute_frameworks = {PyArrowTable}

    link_trekker, trekker = _make_trekker({feature_a.uuid, feature_b.uuid})
    planned_queue: list[Any] = [
        (CrossGroupChildAFG, {feature_a}),
        (CrossGroupChildBFG, {feature_b}),
    ]

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)
    return feature_a, feature_b, link_trekker, trekker


def test_single_orientation_survives_for_shared_link() -> None:
    _, _, link_trekker, trekker = _resolve_cross_group_scenario()
    link = trekker[0]

    assert len(_orientations_present(link_trekker, link)) == 1
    assert len([key for key in link_trekker.data if key[0] == link]) == 1


def test_surviving_orientation_holds_every_trekked_uuid() -> None:
    feature_a, feature_b, link_trekker, trekker = _resolve_cross_group_scenario()
    link = trekker[0]

    orientations = _orientations_present(link_trekker, link)
    assert len(orientations) == 1
    surviving = orientations[0]
    assert link_trekker.data_ordered[surviving] == {feature_a.uuid, feature_b.uuid}
    assert link_trekker.data[surviving] == {feature_a.uuid, feature_b.uuid}


def test_both_groups_are_rewritten_to_the_same_framework() -> None:
    feature_a, feature_b, _, _ = _resolve_cross_group_scenario()

    assert feature_a.compute_frameworks == feature_b.compute_frameworks
    assert feature_a.compute_frameworks == {PyArrowTable}


def test_cross_group_disagreement_raises_at_planning_time() -> None:
    feature_left_only = Feature("cross_group_left_only")
    feature_right_only = Feature("cross_group_right_only")
    feature_left_only.compute_frameworks = {PandasDataFrame}
    feature_right_only.compute_frameworks = {PyArrowTable}

    link_trekker, _ = _make_trekker({feature_left_only.uuid, feature_right_only.uuid})
    planned_queue: list[Any] = [
        (CrossGroupChildAFG, {feature_left_only}),
        (CrossGroupChildBFG, {feature_right_only}),
    ]

    with pytest.raises(ValueError):
        ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)
