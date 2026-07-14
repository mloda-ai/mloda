"""Cross-plugin PROPERTY_MAPPING consistency sweep.

PROPERTY_MAPPING values are ``PropertySpec`` instances (issue #694): the dataclass
constructor enforces the per-spec invariants (known fields, flag types, strict needs a
value space), so this sweep pins what the type system cannot: every shipped plugin spec
IS a ``PropertySpec``, documents itself with a non-empty explanation, and enumerated
value spaces opt into strict validation.
"""

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import is_no_default
from mloda.provider import DefaultOptionKeys, FeatureChainParser, PropertySpec
from mloda.user import Options
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


@pytest.mark.parametrize("plugin_cls", ALL_PLUGINS, ids=lambda c: c.__name__)
class TestPropertyMappingConsistency:
    def test_every_spec_is_a_property_spec(self, plugin_cls: type[Any]) -> None:
        """Raw dict specs are retired: every PROPERTY_MAPPING value is a PropertySpec."""
        mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: list[str] = []
        for prop_key, spec in mapping.items():
            if not isinstance(spec, PropertySpec):
                violations.append(str(prop_key))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING entries that are not PropertySpec instances: {violations}"
        )

    def test_every_spec_has_a_nonempty_explanation(self, plugin_cls: type[Any]) -> None:
        """Every spec documents itself: the explanation is a non-empty string."""
        mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: list[str] = []
        for prop_key, spec in mapping.items():
            if not isinstance(spec.explanation, str) or not spec.explanation.strip():
                violations.append(str(prop_key))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING entries without a non-empty explanation: {violations}"
        )

    def test_enum_specs_have_strict_validation_true(self, plugin_cls: type[Any]) -> None:
        """A spec that enumerates its value space should enforce it.

        Specs with an element_validator delegate validation to the callable, and value
        spaces of fewer than two entries are documentation rather than an enumeration.
        """
        mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        violations: list[str] = []
        for prop_key, spec in mapping.items():
            if spec.element_validator is not None:
                continue
            if spec.allowed_values is None or len(spec.allowed_values) < 2:
                continue
            if spec.strict_validation is not True:
                violations.append(str(prop_key))
        assert violations == [], (
            f"{plugin_cls.__name__} PROPERTY_MAPPING entries with enumerated values "
            f"should have strict_validation=True: {violations}"
        )


# A feature name that string-parses to (None, None) for every plugin, so matching it can only
# succeed through the configuration path. Pinned by test_placeholder_feature_name_forces_the_configuration_path.
CONFIG_FEATURE_NAME = "placeholder"

# A key whose sample value cannot be derived from its own spec. Not None: None and False are
# legitimate option values, so absence has to be its own object.
NOT_CONSTRUCTIBLE: Any = object()

# The only required keys in the shipped plugin set whose value space is free-form, so no sample
# can be read off the spec. Keep this table this small: a growing table means keys are turning
# required.
REQUIRED_SAMPLES: dict[tuple[type[Any], str], Any] = {
    (ClusteringFeatureGroup, ClusteringFeatureGroup.K_VALUE): 3,
    (DimensionalityReductionFeatureGroup, DimensionalityReductionFeatureGroup.DIMENSION): 2,
    (ForecastingFeatureGroup, ForecastingFeatureGroup.HORIZON): 3,
    (TimeWindowFeatureGroup, TimeWindowFeatureGroup.WINDOW_SIZE): 3,
}

# What each plugin declares UNCONDITIONALLY optional today. This pin is an INPUT to the builder
# (see builder_optional_keys), not a mirror of the specs: the builder keeps omitting a pinned key
# even once its spec turns it required, which is what makes the sweep fail instead of quietly
# supplying the key and healing the regression it exists to catch.
EXPECTED_OPTIONAL_KEYS: dict[type[Any], frozenset[str]] = {
    AggregatedFeatureGroup: frozenset(),
    ClusteringFeatureGroup: frozenset({ClusteringFeatureGroup.OUTPUT_PROBABILITIES}),
    MissingValueFeatureGroup: frozenset({"constant_value", "group_by_features"}),
    DimensionalityReductionFeatureGroup: frozenset(
        {
            DimensionalityReductionFeatureGroup.TSNE_MAX_ITER,
            DimensionalityReductionFeatureGroup.TSNE_N_ITER_WITHOUT_PROGRESS,
            DimensionalityReductionFeatureGroup.TSNE_METHOD,
            DimensionalityReductionFeatureGroup.PCA_SVD_SOLVER,
            DimensionalityReductionFeatureGroup.ICA_MAX_ITER,
            DimensionalityReductionFeatureGroup.ISOMAP_N_NEIGHBORS,
        }
    ),
    ForecastingFeatureGroup: frozenset({ForecastingFeatureGroup.OUTPUT_CONFIDENCE_INTERVALS}),
    GeoDistanceFeatureGroup: frozenset(),
    NodeCentralityFeatureGroup: frozenset(
        {NodeCentralityFeatureGroup.GRAPH_TYPE, NodeCentralityFeatureGroup.WEIGHT_COLUMN}
    ),
    EncodingFeatureGroup: frozenset(),
    SklearnPipelineFeatureGroup: frozenset({SklearnPipelineFeatureGroup.PIPELINE_PARAMS}),
    ScalingFeatureGroup: frozenset(),
    TextCleaningFeatureGroup: frozenset(),
    TimeWindowFeatureGroup: frozenset(),
}


def key_name(key: Any) -> str:
    """A spec key is a plain string or a DefaultOptionKeys member."""
    return key.value if isinstance(key, DefaultOptionKeys) else str(key)


def _frozenset_literal(keys: frozenset[str]) -> str:
    """The source line a failing message tells the reader to paste into EXPECTED_OPTIONAL_KEYS."""
    if not keys:
        return "frozenset()"
    return "frozenset({" + ", ".join(repr(key) for key in sorted(keys)) + "})"


def is_optional(spec: PropertySpec) -> bool:
    """Unconditionally optional: declares a default and no predicate.

    A declared ``default=None`` is optional; ``NO_DEFAULT`` is the spec declaring no default at all
    (issue #694), which is what the retired dict form expressed by leaving the key out.
    """
    return not is_no_default(spec.default) and spec.required_when is None


def is_conditional(spec: PropertySpec) -> bool:
    """Required only when its required_when predicate fires."""
    return spec.required_when is not None


def optional_keys(plugin_cls: type[Any]) -> frozenset[str]:
    """Optional according to the spec, i.e. what FeatureChainParser._can_skip_required_check waves through."""
    return frozenset(key_name(key) for key, spec in plugin_cls.PROPERTY_MAPPING.items() if is_optional(spec))


def builder_optional_keys(plugin_cls: type[Any]) -> frozenset[str]:
    """The keys the builder leaves out: optional per the PIN or per the SPEC.

    The union is what makes the sweep catch a flip: a key the pin calls optional stays omitted even
    after its spec quietly turns it required, so matching fails instead of the builder silently
    supplying it. The spec side means a newly optional key joins the sweep before anyone pins it.
    """
    return EXPECTED_OPTIONAL_KEYS.get(plugin_cls, frozenset()) | optional_keys(plugin_cls)


def sample_value(plugin_cls: type[Any], key: Any, spec: PropertySpec) -> Any:
    """Derive a valid value for a key from its own spec, or NOT_CONSTRUCTIBLE."""
    if key == DefaultOptionKeys.in_features:
        return [f"in_feature_{i}" for i in range(plugin_cls.MIN_IN_FEATURES)]

    declared = REQUIRED_SAMPLES.get((plugin_cls, key_name(key)), NOT_CONSTRUCTIBLE)
    if declared is not NOT_CONSTRUCTIBLE:
        return declared

    if not is_no_default(spec.default) and spec.default is not None:
        return spec.default

    if spec.allowed_values:
        return sorted(spec.allowed_values, key=repr)[0]

    if not spec.strict_validation:
        # A key whose default is None and whose value space is free-form (e.g. weight_column).
        # A non-strict spec runs no validation, so any value is legal, and only a value that is
        # really in the option set makes omitting it observable.
        return f"sample_{key_name(key)}"

    return NOT_CONSTRUCTIBLE


def _build(plugin_cls: type[Any], include_optional: bool, omit: Any = None) -> tuple[Options, list[Any]]:
    """The options a configuration-based feature would carry, plus the keys the builder supplied.

    Both are returned from one pass so build_options and needed_keys cannot disagree about which
    conditionals fire. A key goes to context when its spec says so, else to group; Options.get()
    searches both. A supplied key without a sample stays in the key list and out of the Options:
    that gap is what test_every_needed_option_is_constructible reports.
    """
    mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
    leave_out = builder_optional_keys(plugin_cls)
    group: dict[str, Any] = {}
    context: dict[str, Any] = {}
    supplied: list[Any] = []

    def add(key: Any, spec: PropertySpec) -> None:
        if key == omit:
            return
        supplied.append(key)
        value = sample_value(plugin_cls, key, spec)
        if value is NOT_CONSTRUCTIBLE:
            return
        target = context if spec.context else group
        target[key] = value

    for key, spec in mapping.items():
        if key_name(key) not in leave_out and not is_conditional(spec):
            add(key, spec)

    if include_optional:
        for key, spec in mapping.items():
            if key_name(key) in leave_out:
                add(key, spec)

    # Conditionals last, so their predicate reads everything else the builder supplies, as
    # production's predicate reads complete Options. Evaluated one at a time against the growing
    # set: that is what resolves a mutually exclusive pair down to exactly one key.
    for key, spec in mapping.items():
        if key_name(key) in leave_out or not is_conditional(spec):
            continue
        if spec.required_when is not None and spec.required_when(Options(group=dict(group), context=dict(context))):
            add(key, spec)

    return Options(group=group, context=context), supplied


def build_options(plugin_cls: type[Any], include_optional: bool, omit: Any = None) -> Options:
    return _build(plugin_cls, include_optional, omit)[0]


def needed_keys(plugin_cls: type[Any]) -> list[Any]:
    """The keys the builder supplies: everything it does not leave out, plus the conditionals that fire."""
    return _build(plugin_cls, include_optional=False)[1]


OPTIONAL_CASES: list[tuple[type[Any], Any]] = [
    (plugin_cls, key)
    for plugin_cls in ALL_PLUGINS
    for key in plugin_cls.PROPERTY_MAPPING
    if key_name(key) in builder_optional_keys(plugin_cls)
]

OPTIONAL_CASE_IDS: list[str] = [f"{plugin_cls.__name__}-{key_name(key)}" for plugin_cls, key in OPTIONAL_CASES]


class TestOptionalOptionOmission:
    """A configuration-based feature that omits an OPTIONAL option still matches its feature group.

    The cases come from the pin unioned with the specs, so a new optional key joins the sweep on its
    own, while a key that quietly turns required stays omitted and fails matching here.
    """

    @pytest.mark.parametrize("plugin_cls", ALL_PLUGINS, ids=lambda c: c.__name__)
    def test_placeholder_feature_name_forces_the_configuration_path(self, plugin_cls: type[Any]) -> None:
        """Without this, the sweep could pass by string-matching the name and assert nothing."""
        parsed = FeatureChainParser.parse_feature_name(CONFIG_FEATURE_NAME, plugin_cls._get_prefix_patterns())
        assert parsed == (None, None), (
            f"'{CONFIG_FEATURE_NAME}' string-parses to {parsed} for {plugin_cls.__name__}, so matching it "
            f"would short-circuit on the string path and the option sweep below would assert nothing."
        )

    @pytest.mark.parametrize("plugin_cls", ALL_PLUGINS, ids=lambda c: c.__name__)
    def test_every_needed_option_is_constructible(self, plugin_cls: type[Any]) -> None:
        """Every key the builder supplies has a sample, so an unmatched feature below means a real bug."""
        mapping: dict[str, Any] = plugin_cls.PROPERTY_MAPPING
        missing = [
            key_name(key)
            for key in needed_keys(plugin_cls)
            if sample_value(plugin_cls, key, mapping[key]) is NOT_CONSTRUCTIBLE
        ]
        assert missing == [], (
            f"{plugin_cls.__name__} needs option(s) {missing} for which no value can be derived from the spec "
            f"(no default, no allowed_values, strict validation). Either the key should be optional, or add an "
            f"entry to REQUIRED_SAMPLES keyed by (plugin_cls, key)."
        )

    @pytest.mark.parametrize("plugin_cls", ALL_PLUGINS, ids=lambda c: c.__name__)
    def test_config_feature_matches_with_every_optional_option_omitted(self, plugin_cls: type[Any]) -> None:
        """Supplying only the non-optional options is enough to match."""
        options = build_options(plugin_cls, include_optional=False)
        supplied = sorted(key_name(key) for key in options.keys())
        absent = sorted(key_name(key) for key in plugin_cls.PROPERTY_MAPPING if key_name(key) not in supplied)

        assert plugin_cls.match_feature_group_criteria(CONFIG_FEATURE_NAME, options), (
            f"{plugin_cls.__name__} rejects a configuration-based feature that supplies every non-optional option "
            f"and omits every optional one. Supplied: {supplied}. Absent: {absent}. "
            f"Optional keys omitted on purpose: {sorted(builder_optional_keys(plugin_cls))}. "
            f"If one of those keys just became required, it is the regression: a configuration-based user who "
            f"omits it no longer matches."
        )

    @pytest.mark.parametrize(("plugin_cls", "optional_key"), OPTIONAL_CASES, ids=OPTIONAL_CASE_IDS)
    def test_optional_option_can_be_omitted(self, plugin_cls: type[Any], optional_key: Any) -> None:
        """Each optional key, one at a time, can be left out while every other option is present."""
        full = build_options(plugin_cls, include_optional=True)
        assert full.get(optional_key) is not None, (
            f"{plugin_cls.__name__} option '{key_name(optional_key)}' has no sample value, so it is not in the "
            f"option set to begin with and omitting it asserts nothing. Give sample_value a way to build it."
        )

        options = build_options(plugin_cls, include_optional=True, omit=optional_key)

        assert plugin_cls.match_feature_group_criteria(CONFIG_FEATURE_NAME, options), (
            f"{plugin_cls.__name__} rejects a configuration-based feature that omits the optional option "
            f"'{key_name(optional_key)}'. Supplied: {sorted(key_name(key) for key in options.keys())}."
        )

    @pytest.mark.parametrize("plugin_cls", ALL_PLUGINS, ids=lambda c: c.__name__)
    def test_optional_options_match_the_declared_inventory(self, plugin_cls: type[Any]) -> None:
        """The spec-derived optional keys are the ones the inventory pins.

        The pin is what turns a key silently becoming required into a failure here, and it keeps the
        builder omitting that key so the matching sweep above fails too.
        """
        derived = optional_keys(plugin_cls)
        expected = EXPECTED_OPTIONAL_KEYS.get(plugin_cls)

        assert expected is not None, (
            f"{plugin_cls.__name__} has no optional-option inventory. Add this line to EXPECTED_OPTIONAL_KEYS: "
            f"{plugin_cls.__name__}: {_frozenset_literal(derived)},"
        )
        assert derived == expected, (
            f"{plugin_cls.__name__} optional-option inventory drifted. "
            f"Newly optional: {sorted(derived - expected)}. No longer optional: {sorted(expected - derived)}. "
            f"A key that stopped being optional breaks configuration-based features that omit it; "
            f"a newly optional key needs its line updated to {_frozenset_literal(derived)}."
        )
