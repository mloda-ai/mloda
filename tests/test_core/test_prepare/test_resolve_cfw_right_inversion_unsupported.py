"""RIGHT-jointype inversion must not resolve a feature onto a framework it never declared."""

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


class RightInversionLeftFG(FeatureGroup):
    pass


class RightInversionRightFG(FeatureGroup):
    pass


class RightInversionChildFG(FeatureGroup):
    pass


def _make_right_trekker(trekked_uuids: set[UUID]) -> tuple[LinkTrekker, Any]:
    link = Link.right(JoinSpec(RightInversionLeftFG, "idx"), JoinSpec(RightInversionRightFG, "idx"))
    trekker = (link, PandasDataFrame, PyArrowTable)

    link_trekker = LinkTrekker()
    # Production shares one set object between data and data_ordered, and invert_link relies on that.
    shared_uuids = set(trekked_uuids)
    link_trekker.data[trekker] = shared_uuids
    link_trekker.data_ordered[trekker] = shared_uuids
    return link_trekker, trekker


def test_right_join_inversion_onto_undeclared_framework_raises() -> None:
    """Member declares only left_cfw; the RIGHT-branch inversion still resolves it to right_cfw."""
    feature = Feature("right_inversion_feature")
    feature.compute_frameworks = {PandasDataFrame}

    link_trekker, _ = _make_right_trekker({feature.uuid})
    planned_queue: list[Any] = [(RightInversionChildFG, {feature})]

    with pytest.raises(ValueError, match="declares the compute framework") as excinfo:
        ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    message = str(excinfo.value)
    assert str(feature.name) in message, message
    assert PandasDataFrame.get_class_name() in message, message
    assert PyArrowTable.get_class_name() in message, message
