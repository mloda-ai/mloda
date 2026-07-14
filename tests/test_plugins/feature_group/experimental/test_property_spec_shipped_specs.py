"""The builder must reproduce every shipped optional-with-``None``-default spec (issue #733).

Six specs ship with ``default=None``, which makes them OPTIONAL (``_can_skip_required_check``
keys off a DECLARED default, and ``NO_DEFAULT`` is the "none declared" sentinel). Before the
sentinel existed, ``property_spec(..., default=None)`` dropped the default and silently turned
them REQUIRED, so migrating one to the builder was a behavior change. These pins are what would
have caught the ``weight_column`` regression (issue #723).
"""

from __future__ import annotations

import dataclasses

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import PropertySpec, property_spec
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup


def _shipped(feature_group: type[FeatureGroup], key: str) -> PropertySpec:
    """Return the spec a shipped plugin declares for ``key``."""
    mapping = feature_group.PROPERTY_MAPPING
    assert mapping is not None, f"{feature_group.__name__} declares no PROPERTY_MAPPING"
    spec: PropertySpec = mapping[key]
    return spec


# PIPELINE_NAME and PIPELINE_STEPS each carry a required_when predicate ("required only when the
# other is absent"). Two separately written lambdas are never ``==``, so a rebuilt spec can only
# equal the shipped one if the builder is handed the SAME callable object: read the predicate off
# the shipped spec and thread it through by identity.
_PIPELINE_NAME_SHIPPED: PropertySpec = _shipped(SklearnPipelineFeatureGroup, SklearnPipelineFeatureGroup.PIPELINE_NAME)
_PIPELINE_STEPS_SHIPPED: PropertySpec = _shipped(
    SklearnPipelineFeatureGroup, SklearnPipelineFeatureGroup.PIPELINE_STEPS
)


# Shipped specs the builder must reproduce EXACTLY: (id, built, shipped).
_EXACT_CASES: list[tuple[str, PropertySpec, PropertySpec]] = [
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
            required_when=_PIPELINE_STEPS_SHIPPED.required_when,
        ),
        _PIPELINE_STEPS_SHIPPED,
    ),
    (
        "sklearn_pipeline.pipeline_params",
        property_spec("Pipeline parameters dictionary", default=None),
        _shipped(SklearnPipelineFeatureGroup, SklearnPipelineFeatureGroup.PIPELINE_PARAMS),
    ),
]

# Built with its own wording, so it matches the shipped spec on every field BUT ``explanation``.
_PIPELINE_NAME_BUILT: PropertySpec = property_spec(
    "Name of the sklearn pipeline to apply",
    strict=True,
    allowed_values=SklearnPipelineFeatureGroup.PIPELINE_TYPES,
    default=None,
    required_when=_PIPELINE_NAME_SHIPPED.required_when,
)

_ALL_BUILT: list[tuple[str, PropertySpec]] = [(case_id, built) for case_id, built, _ in _EXACT_CASES] + [
    ("sklearn_pipeline.pipeline_name", _PIPELINE_NAME_BUILT)
]

_ALL_SHIPPED: list[tuple[str, PropertySpec]] = [(case_id, shipped) for case_id, _, shipped in _EXACT_CASES] + [
    ("sklearn_pipeline.pipeline_name", _PIPELINE_NAME_SHIPPED)
]


class TestShippedOptionalSpecsAreBuildable:
    """Every shipped optional-with-``None``-default spec is expressible with ``property_spec``."""

    @pytest.mark.parametrize(
        ("built", "shipped"),
        [pytest.param(built, shipped, id=case_id) for case_id, built, shipped in _EXACT_CASES],
    )
    def test_builder_reproduces_shipped_spec_exactly(self, built: PropertySpec, shipped: PropertySpec) -> None:
        """The built spec is field-for-field the spec the plugin ships."""
        assert built == shipped

    def test_builder_reproduces_pipeline_name_up_to_the_explanation(self) -> None:
        """PIPELINE_NAME matches once the wording of ``explanation`` is set aside.

        The two are worded differently, which is no behavior change: no core rule reads
        ``explanation``. Every OTHER field must agree.
        """
        normalized = dataclasses.replace(_PIPELINE_NAME_BUILT, explanation=_PIPELINE_NAME_SHIPPED.explanation)

        assert normalized == _PIPELINE_NAME_SHIPPED
        assert _PIPELINE_NAME_BUILT.explanation != _PIPELINE_NAME_SHIPPED.explanation

    @pytest.mark.parametrize(
        "built",
        [pytest.param(built, id=case_id) for case_id, built in _ALL_BUILT],
    )
    def test_built_spec_is_optional(self, built: PropertySpec) -> None:
        """Each built spec stays OPTIONAL: migration must not make the option required."""
        assert FeatureChainParser._can_skip_required_check(built) is True

    @pytest.mark.parametrize(
        "shipped",
        [pytest.param(shipped, id=case_id) for case_id, shipped in _ALL_SHIPPED],
    )
    def test_shipped_spec_is_optional(self, shipped: PropertySpec) -> None:
        """Each SHIPPED spec is OPTIONAL: the invariant #723 broke, pinned on what plugins declare.

        ``test_built_spec_is_optional`` only checks specs this test builds. Now that these plugins
        declare ``PropertySpec`` directly, the shipped side needs its own pin.
        """
        assert FeatureChainParser._can_skip_required_check(shipped) is True
