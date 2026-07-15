"""Pin ``FeatureGroup.options_with_defaults``: materialize declared PropertySpec defaults into Options (#766).

The mechanism returns an Options equal to the input PLUS, for every PROPERTY_MAPPING key that is absent
(``options.get(key) is None``) and declares a concrete default (``not is_no_default(spec.default)`` and
``spec.default is not None``), the value ``spec.default`` placed in ``.context`` when ``spec.context`` else
``.group``. A present value (even a falsy 0/""/False) is never overridden, and the input is never mutated.

Group 3 (``resolve_subtype`` delegating to this mechanism) is already pinned by
``tests/test_core/test_prepare/test_subtype_option_parity.py::TestSbparPropertyMappingDefaultParity::test_resolve_subtype_applies_declared_default``
(empty options -> declared default subtype), so it is referenced here rather than duplicated.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import is_no_default
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import PropertySpec
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    The tests below define FeatureGroup subclasses to exercise the author experience.
    Those class objects sit in reference cycles, so they linger in
    ``FeatureGroup.__subclasses__()`` until a GC cycle runs. While they linger, other
    tests that enumerate feature groups via ``get_all_subclasses(FeatureGroup)`` trip
    over them. After each test we force a collection to reclaim the now-unreferenced
    classes and assert that none of this module's classes remain registered, pinning
    the no-pollution contract.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


# PROPERTY_MAPPING keys for the throwaway feature group, one per default edge under test.
CTX_KEY = "owd_ctx_default"  # context concrete default
GRP_KEY = "owd_grp_default"  # group concrete default (context=False)
FALSY_KEY = "owd_falsy_default"  # non-strict default is a non-falsy string; set to a falsy present value
NONE_KEY = "owd_none_default"  # declared default=None (optional, applies no value)
REQ_KEY = "owd_required"  # NO_DEFAULT (required by omission)


def _options_with_defaults(fg: type[FeatureGroup], options: Options) -> Any:
    """Invoke ``FeatureGroup.options_with_defaults`` directly; the method now lives on the base class.

    A direct typed call (``fg`` is ``type[FeatureGroup]``, the classmethod is declared on ``FeatureGroup``)
    stays ``mypy --strict`` clean and retires the dynamic ``getattr`` seam the RED phase needed while the
    method was still absent.
    """
    return fg.options_with_defaults(options)


def _declared_default(fg: type[FeatureGroup], key: str) -> Any:
    """The default declared for ``key`` in ``fg``'s PROPERTY_MAPPING."""
    mapping: dict[str, PropertySpec] = fg.PROPERTY_MAPPING or {}
    return mapping[key].default


def _make_probe_fg() -> type[FeatureGroup]:
    """A throwaway FeatureGroup whose PROPERTY_MAPPING exercises every default edge (no required_when)."""

    class ProbeDefaultsFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            CTX_KEY: PropertySpec("A context concrete default.", context=True, default="ctx_val"),
            GRP_KEY: PropertySpec("A group concrete default.", context=False, default="grp_val"),
            FALSY_KEY: PropertySpec("Non-strict default is a non-falsy string.", context=True, default="nonfalsy"),
            NONE_KEY: PropertySpec("Optional: a declared None applies no value.", context=True, default=None),
            REQ_KEY: PropertySpec("Required by omission (NO_DEFAULT).", context=True),
        }

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return data

    return ProbeDefaultsFeatureGroup


class TestOptionsWithDefaultsMechanism:
    """FeatureGroup.options_with_defaults fills declared concrete defaults into runtime Options."""

    def test_absent_context_default_filled_into_context(self) -> None:
        """An absent key with a context concrete default lands in ``.context`` with that value."""
        fg = _make_probe_fg()
        filled = _options_with_defaults(fg, Options())
        assert filled.get(CTX_KEY) == "ctx_val"
        assert CTX_KEY in filled.context
        assert CTX_KEY not in filled.group

    def test_absent_group_default_filled_into_group(self) -> None:
        """An absent ``context=False`` key with a concrete default lands in ``.group`` with that value."""
        fg = _make_probe_fg()
        filled = _options_with_defaults(fg, Options())
        assert filled.get(GRP_KEY) == "grp_val"
        assert GRP_KEY in filled.group
        assert GRP_KEY not in filled.context

    def test_present_value_overrides_declared_default(self) -> None:
        """An explicit value wins over the declared default."""
        fg = _make_probe_fg()
        filled = _options_with_defaults(fg, Options(context={CTX_KEY: "explicit"}))
        assert filled.get(CTX_KEY) == "explicit"

    @pytest.mark.parametrize("falsy", [0, "", False])
    def test_present_falsy_value_is_kept_not_defaulted(self, falsy: Any) -> None:
        """A present falsy value (0/""/False) is kept; the default must not override it (``is None`` not ``or``)."""
        fg = _make_probe_fg()
        filled = _options_with_defaults(fg, Options(context={FALSY_KEY: falsy}))
        assert filled.get(FALSY_KEY) == falsy
        assert filled.get(FALSY_KEY) != "nonfalsy"

    def test_default_none_key_is_not_filled(self) -> None:
        """A ``default=None`` key stays absent from both group and context."""
        fg = _make_probe_fg()
        filled = _options_with_defaults(fg, Options())
        assert filled.get(NONE_KEY) is None
        assert NONE_KEY not in filled.group
        assert NONE_KEY not in filled.context

    def test_required_no_default_key_is_not_filled(self) -> None:
        """A NO_DEFAULT (required) key is not filled."""
        fg = _make_probe_fg()
        filled = _options_with_defaults(fg, Options())
        assert filled.get(REQ_KEY) is None
        assert REQ_KEY not in filled.group
        assert REQ_KEY not in filled.context

    def test_input_options_never_mutated(self) -> None:
        """The input Options is not mutated; no default leaks back into it."""
        fg = _make_probe_fg()
        original = Options()
        group_before = dict(original.group)
        context_before = dict(original.context)
        _options_with_defaults(fg, original)
        assert original.group == group_before
        assert original.context == context_before
        assert CTX_KEY not in original.context
        assert GRP_KEY not in original.group

    def test_returns_options_and_preserves_propagate_context_keys(self) -> None:
        """Returns an Options; ``propagate_context_keys`` survives the call."""
        fg = _make_probe_fg()
        opts = Options(context={CTX_KEY: "x"}, propagate_context_keys=frozenset({CTX_KEY}))
        filled = _options_with_defaults(fg, opts)
        assert isinstance(filled, Options)
        assert filled.propagate_context_keys == frozenset({CTX_KEY})

    def test_fill_path_preserves_forwarding_provenance(self) -> None:
        """The filled view keeps inherited/forwarded provenance; a public Options must not silently drop it."""
        fg = _make_probe_fg()
        opts = Options(context={"seed": "v"})
        opts.inherited_group_keys = frozenset({"a"})
        opts.inherited_context_keys = frozenset({"b"})
        opts.last_forwarded_group_keys = frozenset({"c"})
        filled = _options_with_defaults(fg, opts)
        assert filled.get(CTX_KEY) == "ctx_val"  # a default WAS filled, so this is the fresh-Options path
        assert filled.inherited_group_keys == frozenset({"a"})
        assert filled.inherited_context_keys == frozenset({"b"})
        assert filled.last_forwarded_group_keys == frozenset({"c"})


# Shipped feature groups (mirrors test_property_mapping_consistency.ALL_PLUGINS) for the drift sweep.
ALL_PLUGINS: list[type[Any]] = [
    AggregatedFeatureGroup,
    ClusteringFeatureGroup,
    MissingValueFeatureGroup,
    DimensionalityReductionFeatureGroup,
    ForecastingFeatureGroup,
    GeoDistanceFeatureGroup,
    NodeCentralityFeatureGroup,
    EncodingFeatureGroup,
    SklearnPipelineFeatureGroup,
    ScalingFeatureGroup,
    TextCleaningFeatureGroup,
    TimeWindowFeatureGroup,
]


class TestShippedDefaultsMaterialize:
    """Shipped feature groups' declared concrete defaults materialize verbatim (issue #766 DoD)."""

    def test_node_centrality_graph_type_default(self) -> None:
        """NodeCentralityFeatureGroup.graph_type materializes to its declared 'undirected'."""
        key = NodeCentralityFeatureGroup.GRAPH_TYPE
        filled = _options_with_defaults(NodeCentralityFeatureGroup, Options())
        assert filled.get(key) == _declared_default(NodeCentralityFeatureGroup, key) == "undirected"

    def test_clustering_output_probabilities_default(self) -> None:
        """ClusteringFeatureGroup.output_probabilities materializes to its declared False (a falsy default)."""
        key = ClusteringFeatureGroup.OUTPUT_PROBABILITIES
        filled = _options_with_defaults(ClusteringFeatureGroup, Options())
        assert filled.get(key) is False
        assert _declared_default(ClusteringFeatureGroup, key) is False

    def test_forecasting_output_confidence_intervals_default(self) -> None:
        """ForecastingFeatureGroup.output_confidence_intervals materializes to its declared False."""
        key = ForecastingFeatureGroup.OUTPUT_CONFIDENCE_INTERVALS
        filled = _options_with_defaults(ForecastingFeatureGroup, Options())
        assert filled.get(key) is False
        assert _declared_default(ForecastingFeatureGroup, key) is False

    def test_dimensionality_reduction_defaults(self) -> None:
        """All six DimensionalityReductionFeatureGroup algorithm defaults materialize verbatim."""
        fg = DimensionalityReductionFeatureGroup
        filled = _options_with_defaults(fg, Options())
        expected: dict[str, Any] = {
            fg.TSNE_MAX_ITER: 250,
            fg.TSNE_N_ITER_WITHOUT_PROGRESS: 50,
            fg.TSNE_METHOD: "barnes_hut",
            fg.PCA_SVD_SOLVER: "auto",
            fg.ICA_MAX_ITER: 200,
            fg.ISOMAP_N_NEIGHBORS: 5,
        }
        for key, value in expected.items():
            assert filled.get(key) == _declared_default(fg, key) == value

    def test_every_shipped_concrete_default_materializes(self) -> None:
        """Every shipped concrete (non-None, declared) default materializes to itself.

        Iterating every key future-proofs any newly added concrete default across the shipped set.
        """
        checked = 0
        for plugin_cls in ALL_PLUGINS:
            mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING or {}
            filled = _options_with_defaults(plugin_cls, Options())
            for key, spec in mapping.items():
                if is_no_default(spec.default) or spec.default is None:
                    continue
                assert filled.get(key) == spec.default, (
                    f"{plugin_cls.__name__}.{key}: options_with_defaults yielded {filled.get(key)!r}, "
                    f"expected the declared default {spec.default!r}."
                )
                checked += 1
        assert checked > 0, "the sweep asserted nothing; no shipped concrete defaults were discovered."


class TestMutableDefaultIsolation:
    """A materialized mutable default must be a per-call copy, so compute-time mutation cannot corrupt the spec."""

    def test_materialized_mutable_default_is_isolated(self) -> None:
        """A mutable concrete default is copied per call; compute-time mutation cannot corrupt the spec."""

        class MutableDefaultFeatureGroup(FeatureGroup):
            PROPERTY_MAPPING = {"items": PropertySpec("A mutable default.", context=True, default=[])}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        materialized = _options_with_defaults(MutableDefaultFeatureGroup, Options()).get("items")
        assert materialized == []
        materialized.append("mutated")
        assert MutableDefaultFeatureGroup.PROPERTY_MAPPING["items"].default == [], "the declared default was corrupted"
        assert _options_with_defaults(MutableDefaultFeatureGroup, Options()).get("items") == [], (
            "a later call saw the leak"
        )
