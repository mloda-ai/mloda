"""The builder must reproduce every shipped optional-with-``None``-default spec (issue #733).

Six specs ship hand-written today with ``default: None``, which makes them OPTIONAL
(``_can_skip_required_check`` keys off the PRESENCE of the ``default`` key). Until the
``NO_DEFAULT`` sentinel exists, ``property_spec(..., default=None)`` drops that key and
silently turns them REQUIRED, so migrating one to the builder is a behavior change. These
pins are what would have caught the ``weight_column`` regression (issue #723).
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DefaultOptionKeys, property_spec
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup


def _shipped(feature_group: type[FeatureGroup], key: str) -> dict[str, Any]:
    """Return the hand-written spec a shipped plugin declares for ``key``."""
    mapping = feature_group.PROPERTY_MAPPING
    assert mapping is not None, f"{feature_group.__name__} declares no PROPERTY_MAPPING"
    spec: dict[str, Any] = mapping[key]
    return spec


# PIPELINE_NAME and PIPELINE_STEPS each carry a required_when predicate ("required only when the
# other is absent"). Two separately written lambdas are never ``==``, so a rebuilt spec can only
# equal the shipped one if the builder is handed the SAME callable object: read the predicate off
# the shipped spec and thread it through by identity.
_PIPELINE_NAME_SHIPPED: dict[str, Any] = _shipped(
    SklearnPipelineFeatureGroup, SklearnPipelineFeatureGroup.PIPELINE_NAME
)
_PIPELINE_STEPS_SHIPPED: dict[str, Any] = _shipped(
    SklearnPipelineFeatureGroup, SklearnPipelineFeatureGroup.PIPELINE_STEPS
)


# Shipped specs the builder must reproduce EXACTLY: (id, built, shipped).
_EXACT_CASES: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    (
        "node_centrality.weight_column",
        property_spec("Column name for edge weights (optional)", default=None),
        _shipped(NodeCentralityFeatureGroup, NodeCentralityFeatureGroup.WEIGHT_COLUMN),
    ),
    (
        "missing_value.constant_value",
        property_spec("Constant value to use for constant imputation method", default=None),
        _shipped(MissingValueFeatureGroup, "constant_value"),
    ),
    (
        "missing_value.group_by_features",
        property_spec("Optional list of features to group by before imputation", default=None),
        _shipped(MissingValueFeatureGroup, "group_by_features"),
    ),
    (
        "sklearn_pipeline.pipeline_steps",
        property_spec(
            "List of pipeline steps as (name, transformer) tuples",
            default=None,
            required_when=_PIPELINE_STEPS_SHIPPED[DefaultOptionKeys.required_when],
        ),
        _PIPELINE_STEPS_SHIPPED,
    ),
    (
        "sklearn_pipeline.pipeline_params",
        property_spec("Pipeline parameters dictionary", default=None),
        _shipped(SklearnPipelineFeatureGroup, SklearnPipelineFeatureGroup.PIPELINE_PARAMS),
    ),
]

# PIPELINE_NAME ships without an "explanation" key, which the builder always emits.
_PIPELINE_NAME_BUILT: dict[str, Any] = property_spec(
    "Name of the sklearn pipeline to apply",
    strict=True,
    allowed_values=SklearnPipelineFeatureGroup.PIPELINE_TYPES,
    default=None,
    required_when=_PIPELINE_NAME_SHIPPED[DefaultOptionKeys.required_when],
)

_ALL_BUILT: list[tuple[str, dict[str, Any]]] = [(case_id, built) for case_id, built, _ in _EXACT_CASES] + [
    ("sklearn_pipeline.pipeline_name", _PIPELINE_NAME_BUILT)
]

_ALL_SHIPPED: list[tuple[str, dict[str, Any]]] = [(case_id, shipped) for case_id, _, shipped in _EXACT_CASES] + [
    ("sklearn_pipeline.pipeline_name", _PIPELINE_NAME_SHIPPED)
]


class TestShippedOptionalSpecsAreBuildable:
    """Every shipped optional-with-``None``-default spec is expressible with ``property_spec``."""

    @pytest.mark.parametrize(
        ("built", "shipped"),
        [pytest.param(built, shipped, id=case_id) for case_id, built, shipped in _EXACT_CASES],
    )
    def test_builder_reproduces_shipped_spec_exactly(self, built: dict[str, Any], shipped: dict[str, Any]) -> None:
        """The built spec is byte-for-byte the hand-written dict the plugin ships."""
        assert built == shipped

    def test_builder_reproduces_pipeline_name_up_to_added_explanation(self) -> None:
        """PIPELINE_NAME matches once the builder's explanation is set aside.

        Migrating it ADDS the "explanation" key the shipped dict lacks: an improvement, not a
        behavior change (no core rule reads "explanation").
        """
        without_explanation = {key: value for key, value in _PIPELINE_NAME_BUILT.items() if key != "explanation"}

        assert without_explanation == _PIPELINE_NAME_SHIPPED

    @pytest.mark.parametrize(
        "built",
        [pytest.param(built, id=case_id) for case_id, built in _ALL_BUILT],
    )
    def test_built_spec_is_optional(self, built: dict[str, Any]) -> None:
        """Each built spec stays OPTIONAL: migration must not make the option required."""
        assert FeatureChainParser._can_skip_required_check(built) is True

    @pytest.mark.parametrize(
        "shipped",
        [pytest.param(shipped, id=case_id) for case_id, shipped in _ALL_SHIPPED],
    )
    def test_shipped_spec_is_optional(self, shipped: dict[str, Any]) -> None:
        """Each SHIPPED spec is OPTIONAL: the invariant #723 broke, pinned on what plugins declare.

        ``test_built_spec_is_optional`` only checks specs this test builds. Once these plugins
        migrate to the builder, ``built == shipped`` degrades into comparing builder output with
        builder output, so the shipped side needs its own pin.
        """
        assert FeatureChainParser._can_skip_required_check(shipped) is True
