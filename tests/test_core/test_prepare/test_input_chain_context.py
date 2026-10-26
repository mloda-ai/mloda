"""Chain attempts use the same request constraints, forwarding and declarations as final expansion."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import FeatureGroup, FeatureSet, PropertySpec, escalate_match_abort
from mloda.user import DataAccessCollection, Feature, FeatureName, Index, JoinSpec, Link, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_prepare.input_chain_helpers import (
    CHOICE,
    LEAF,
    MIDDLE,
    MISSING,
    PROBE_CALLS,
    RAW_A,
    ChainA2320,
    ChainB2320,
    ChainChecked2320,
    ChainGate2320,
    ChainLeaf2320,
    ChainMiddle2320,
    ChainOtherLeaf2320,
    ChainStub2320,
    chain_access,
    prepare_chain,
    result_for,
)


OPTION_KEY = "ticket_chain_2320"
OPTION_VALUE = "ready_chain_2320"
DOMAIN = "domain_chain_2320"
OTHER_DOMAIN = "other_domain_chain_2320"
PART_BASE = "parts_chain_2320"
RENAMED = "renamed_choice_chain_2320"
DEFAULT_KEY = "default_chain_2320"
DEFAULT_VALUE = "materialized_chain_2320"


class ChainOptionLeaf2320(ChainLeaf2320):
    """Only matches with the forwarded option value, whether in group or context."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return super().match_feature_group_criteria(feature_name, options, data_access_collection) and (
            options.get(OPTION_KEY) == OPTION_VALUE
        )


class ChainParts2320(ChainStub2320):
    """Both named and digit output parts depend on the same leaf."""

    MATCHED_NAMES = frozenset({PART_BASE})
    SUPPORTED_NAMES = frozenset({PART_BASE})
    OUTPUT_PARTS = ("value", "0")
    INPUTS = (LEAF,)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        if str(feature_name) != PART_BASE:
            raise ValueError("part dependencies must be resolved through the base name")
        return super().input_features(options, feature_name)


class ChainRenaming2320(ChainStub2320):
    """Dependency declaration sees the name produced by set_feature_name."""

    MATCHED_NAMES = frozenset({CHOICE})

    def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
        return FeatureName(RENAMED)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        if str(feature_name) != RENAMED:
            raise ValueError("dependencies need the normalized name")
        return {Feature(LEAF)}


class ChainDefault2320(ChainStub2320):
    """Declared options choose a viable input before intake adds the property's default."""

    MATCHED_NAMES = frozenset({CHOICE})
    PROPERTY_MAPPING = {
        DEFAULT_KEY: PropertySpec("Selects the declared dependency.", context=True, default=DEFAULT_VALUE)
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        return {Feature(LEAF if options.get(DEFAULT_KEY) is None else MISSING)}


class ChainDiamond2320(ChainStub2320):
    """Two different middle features share the same source input."""

    MATCHED_NAMES = frozenset({LEAF, MIDDLE})
    INPUTS = (RAW_A,)


@pytest.mark.parametrize("valid_first", [True, False], ids=["valid-first", "invalid-first"])
@pytest.mark.parametrize("constraint", ["scope", "declarations", "domain", "framework", "group", "context"])
def test_equal_input_names_keep_each_candidates_own_resolution_constraints(
    constraint: str, valid_first: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    groups: set[type[FeatureGroup]] = {ChainA2320, ChainB2320, ChainLeaf2320}
    if constraint == "scope":
        groups.add(ChainOtherLeaf2320)
        valid = Feature(LEAF, feature_group=ChainLeaf2320)
        invalid = Feature(LEAF, feature_group=ChainOtherLeaf2320)
    elif constraint == "declarations":

        def declared(cls: type[ChainLeaf2320], features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
            return {"unit_chain_2320": "known"}

        monkeypatch.setattr(ChainLeaf2320, "declared_attributes", classmethod(declared))
        valid = Feature(LEAF, required_declarations={"unit_chain_2320": "known"})
        invalid = Feature(LEAF, required_declarations={"unit_chain_2320": "wrong"})
    elif constraint == "domain":
        monkeypatch.setattr(ChainLeaf2320, "DOMAIN_NAME", DOMAIN)
        valid = Feature(LEAF, domain=DOMAIN)
        invalid = Feature(LEAF, domain=OTHER_DOMAIN)
    elif constraint == "framework":
        valid = Feature(LEAF, compute_framework=PyArrowTable.__name__)
        invalid = Feature(LEAF, compute_framework=PythonDictFramework.__name__)
    else:
        groups.remove(ChainLeaf2320)
        groups.add(ChainOptionLeaf2320)
        valid_options = Options(group={OPTION_KEY: OPTION_VALUE})
        invalid_options = Options(group={OPTION_KEY: "wrong"})
        if constraint == "context":
            valid_options = Options(context={OPTION_KEY: OPTION_VALUE})
            invalid_options = Options(context={OPTION_KEY: "wrong"})
        valid = Feature(LEAF, options=valid_options)
        invalid = Feature(LEAF, options=invalid_options)
    monkeypatch.setattr(ChainA2320, "INPUTS", (valid if valid_first else invalid,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (invalid if valid_first else valid,))
    ordered_groups: list[type[FeatureGroup]] = [
        ChainA2320,
        ChainB2320,
        *sorted(groups - {ChainA2320, ChainB2320}, key=lambda group: group.__name__),
    ]
    plugins: FeatureGroupEnvironmentMapping = {group: {PyArrowTable} for group in ordered_groups}

    with run_match_cache():
        result = IdentifyFeatureGroupClass.evaluate(Feature(CHOICE), plugins, None, None)

    winner = ChainA2320 if valid_first else ChainB2320
    loser = ChainB2320 if valid_first else ChainA2320
    assert set(result.identified) == {winner}
    assert result.eliminations[loser].stage == "inputs"
    assert LEAF in result.eliminations[loser].reason


@pytest.mark.parametrize("valid_first", [True, False], ids=["valid-first", "invalid-first"])
def test_same_run_cache_distinguishes_requested_context_options(
    valid_first: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def inputs(self: ChainStub2320, options: Options, name: FeatureName) -> set[Feature]:
        return {Feature(LEAF if options.get(OPTION_KEY) == OPTION_VALUE else MISSING)}

    monkeypatch.setattr(ChainA2320, "input_features", inputs)
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))
    plugins: FeatureGroupEnvironmentMapping = {
        group: {PyArrowTable} for group in (ChainA2320, ChainB2320, ChainLeaf2320)
    }
    order = (OPTION_VALUE, "wrong") if valid_first else ("wrong", OPTION_VALUE)

    with run_match_cache():
        for value in order:
            feature = Feature(CHOICE, options=Options(context={OPTION_KEY: value}))
            result = IdentifyFeatureGroupClass.evaluate(feature, plugins, None, None)
            assert set(result.identified) == ({ChainA2320} if value == OPTION_VALUE else set())


@pytest.mark.parametrize(
    "forwarding", ["all", "allowlist", "exclude-other", "off", "exclude-key", "pull", "no-pull", "push", "cascade"]
)
def test_chain_preflight_uses_the_real_input_option_forwarding_rules(
    forwarding: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = Options(group={OPTION_KEY: OPTION_VALUE, "unrelated_chain_2320": 1})
    input_feature = Feature(LEAF)
    groups = {ChainA2320, ChainB2320, ChainOptionLeaf2320, ChainMiddle2320}
    if forwarding == "allowlist":
        input_feature = Feature(LEAF, forward_group=frozenset({OPTION_KEY}))
    elif forwarding == "exclude-other":
        input_feature = Feature(LEAF, forward_group_exclude={"unrelated_chain_2320"})
    elif forwarding == "off":
        input_feature = Feature(LEAF, forward_group=False)
    elif forwarding == "exclude-key":
        input_feature = Feature(LEAF, forward_group_exclude={OPTION_KEY})
    elif forwarding in {"pull", "no-pull", "push", "cascade"}:
        options = Options(context={OPTION_KEY: OPTION_VALUE})
        if forwarding == "pull":
            input_feature = Feature(LEAF, inherit_context_keys={OPTION_KEY})
        elif forwarding == "push":
            options.propagate_context_keys = frozenset({OPTION_KEY})
        elif forwarding == "cascade":
            options.cascade_context_keys = frozenset({OPTION_KEY})
            input_feature = Feature(MIDDLE)
            monkeypatch.setattr(ChainMiddle2320, "INPUTS", (LEAF,))
    monkeypatch.setattr(ChainA2320, "INPUTS", (input_feature,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))
    request = Feature(CHOICE, options=options)

    if forwarding in {"off", "exclude-key", "no-pull"}:
        result = IdentifyFeatureGroupClass.evaluate(request, {group: {PyArrowTable} for group in groups}, None, None)
        assert result.failure_kind == "none"
        assert result.eliminations[ChainA2320].stage == "inputs"
    else:
        session = prepare_chain(groups, feature=request)
        assert set(result_for(session.resolution_report()).identified) == {ChainA2320}
        assert session.engine is not None
        (leaf,) = session.engine.feature_group_collection[ChainOptionLeaf2320]
        assert leaf.options.get(OPTION_KEY) == OPTION_VALUE
        if forwarding in {"allowlist", "exclude-other"}:
            assert leaf.options.get("unrelated_chain_2320") is None


def test_input_chain_inherits_the_requested_parent_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    for group in (ChainA2320, ChainB2320, ChainLeaf2320):
        monkeypatch.setattr(group, "DOMAIN_NAME", DOMAIN)
    monkeypatch.setattr(ChainOtherLeaf2320, "DOMAIN_NAME", OTHER_DOMAIN)
    monkeypatch.setattr(ChainOtherLeaf2320, "INPUTS", None)
    monkeypatch.setattr(ChainA2320, "INPUTS", (LEAF,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))

    session = prepare_chain(
        {ChainA2320, ChainB2320, ChainLeaf2320, ChainOtherLeaf2320}, feature=Feature(CHOICE, domain=DOMAIN)
    )

    assert set(result_for(session.resolution_report()).identified) == {ChainA2320}
    assert set(result_for(session.resolution_report(), LEAF).identified) == {ChainLeaf2320}
    assert session.engine is not None
    (leaf,) = session.engine.feature_group_collection[ChainLeaf2320]
    assert leaf.domain is not None
    assert leaf.domain.name == DOMAIN


@pytest.mark.parametrize("part", ["value", "0"], ids=["named-part", "digit-part"])
def test_input_output_parts_keep_the_declared_column_and_resolve_the_base_chain(
    part: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_part = f"{PART_BASE}~{part}"
    monkeypatch.setattr(ChainA2320, "INPUTS", (MISSING,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (requested_part,))

    session = prepare_chain({ChainA2320, ChainB2320, ChainParts2320, ChainLeaf2320})

    assert set(result_for(session.resolution_report()).identified) == {ChainB2320}
    assert session.engine is not None
    (consumer,) = session.engine.feature_group_collection[ChainB2320]
    (producer,) = session.engine.feature_group_collection[ChainParts2320]
    assert str(producer.name) == PART_BASE
    assert session.engine.input_feature_columns[consumer.uuid] == (requested_part,)
    assert requested_part in session.engine.resolved_input_feature_names[consumer.uuid]


def test_chain_declaration_observes_the_normalized_feature_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (MISSING,))

    session = prepare_chain({ChainA2320, ChainRenaming2320, ChainLeaf2320})

    assert set(result_for(session.resolution_report()).identified) == {ChainRenaming2320}
    assert session.engine is not None
    (renamed,) = session.engine.feature_group_collection[ChainRenaming2320]
    assert str(renamed.name) == RENAMED
    assert session.engine.input_feature_columns[renamed.uuid] == (LEAF,)


def test_chain_declaration_observes_declared_options_before_default_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (MISSING,))

    session = prepare_chain({ChainA2320, ChainDefault2320, ChainLeaf2320})

    assert set(result_for(session.resolution_report()).identified) == {ChainDefault2320}
    assert session.engine is not None
    (consumer,) = session.engine.feature_group_collection[ChainDefault2320]
    (leaf,) = session.engine.feature_group_collection[ChainLeaf2320]
    assert consumer.options.get(DEFAULT_KEY) == DEFAULT_VALUE
    assert leaf.options.get(DEFAULT_KEY) is None
    assert session.engine.input_feature_columns[consumer.uuid] == (LEAF,)


def test_cached_chain_outcomes_do_not_cross_accessible_plugin_universes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (LEAF,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))
    full: FeatureGroupEnvironmentMapping = {group: {PyArrowTable} for group in (ChainA2320, ChainB2320, ChainLeaf2320)}
    restricted: FeatureGroupEnvironmentMapping = {group: {PyArrowTable} for group in (ChainA2320, ChainB2320)}

    with run_match_cache():
        resolved = IdentifyFeatureGroupClass.evaluate(Feature(CHOICE), full, None, None)
        failed = IdentifyFeatureGroupClass.evaluate(Feature(CHOICE), restricted, None, None)

    assert set(resolved.identified) == {ChainA2320}
    assert failed.failure_kind == "none"
    assert failed.eliminations[ChainA2320].stage == "inputs"


def test_cached_chain_outcomes_do_not_cross_data_access_collections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))
    plugins: FeatureGroupEnvironmentMapping = {
        group: {PyArrowTable} for group in (ChainA2320, ChainB2320, ChainChecked2320)
    }

    with run_match_cache():
        resolved = IdentifyFeatureGroupClass.evaluate(Feature(CHOICE), plugins, None, chain_access(**{RAW_A: [1]}))
        failed = IdentifyFeatureGroupClass.evaluate(Feature(CHOICE), plugins, None, chain_access())

    assert set(resolved.identified) == {ChainA2320}
    assert failed.failure_kind == "none"
    assert failed.eliminations[ChainA2320].stage == "inputs"


def test_cached_chain_outcomes_use_the_current_links(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (LEAF,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))

    def indices(cls: type[ChainLeaf2320]) -> list[Index]:
        return [Index(("id_chain_2320",))]

    monkeypatch.setattr(ChainLeaf2320, "index_columns", classmethod(indices))
    good = Link("inner", JoinSpec(ChainLeaf2320, "id_chain_2320"), JoinSpec(ChainB2320, "id_chain_2320"))
    bad = Link("inner", JoinSpec(ChainLeaf2320, "wrong_chain_2320"), JoinSpec(ChainB2320, "wrong_chain_2320"))
    plugins: FeatureGroupEnvironmentMapping = {
        group: {PyArrowTable} for group in (ChainA2320, ChainB2320, ChainLeaf2320)
    }

    with run_match_cache():
        resolved = IdentifyFeatureGroupClass.evaluate(Feature(CHOICE), plugins, {good}, None)
        failed = IdentifyFeatureGroupClass.evaluate(Feature(CHOICE), plugins, {bad}, None)

    assert set(resolved.identified) == {ChainA2320}
    assert failed.failure_kind == "none"
    assert failed.eliminations[ChainA2320].stage == "inputs"


def test_diamond_dependencies_reuse_the_source_probe_through_final_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (MISSING,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (LEAF, MIDDLE))
    PROBE_CALLS.clear()
    gate = ChainGate2320()

    session = prepare_chain(
        {ChainA2320, ChainB2320, ChainDiamond2320, ChainChecked2320},
        access=chain_access(**{RAW_A: [1]}),
        extender=gate,
    )

    assert set(result_for(session.resolution_report()).identified) == {ChainB2320}
    assert len(gate.calls_for(ChainChecked2320, RAW_A)) == 1
    assert PROBE_CALLS[ChainChecked2320, RAW_A] == 1
    assert session.engine is not None
    assert len(session.engine.feature_group_collection[ChainChecked2320]) == 1
    assert {str(feature.name) for feature in session.engine.feature_group_collection[ChainDiamond2320]} == {
        LEAF,
        MIDDLE,
    }
    (source,) = session.engine.feature_group_collection[ChainChecked2320]
    assert source.input_data_match is not None
    assert source.input_data_match[0] is ChainChecked2320


@pytest.mark.parametrize("as_string", [False, True], ids=["class", "class-name"])
def test_refusal_inside_a_scoped_input_is_contained_to_that_candidate(
    as_string: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = ChainChecked2320.__name__ if as_string else ChainChecked2320
    monkeypatch.setattr(ChainA2320, "INPUTS", (Feature(RAW_A, feature_group=scope),))
    gate = ChainGate2320({RAW_A}, ChainChecked2320)

    session = prepare_chain(
        {ChainA2320, ChainB2320, ChainLeaf2320, ChainChecked2320},
        access=chain_access(**{RAW_A: [1]}),
        extender=gate,
    )

    result = result_for(session.resolution_report())
    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainA2320].stage == "inputs"
    assert "refused input" in result.eliminations[ChainA2320].reason
    assert len(gate.errors) == 1
    assert len(gate.calls_for(ChainChecked2320, RAW_A)) == 1


@pytest.mark.parametrize("marked", [False, True], ids=["ordinary-error", "match-abort"])
def test_input_declaration_errors_are_contained_to_the_attempted_candidate(
    marked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ValueError("invalid input declaration_chain_2320")
    if marked:
        escalate_match_abort(error)

    def inputs(self: ChainStub2320, options: Options, name: FeatureName) -> set[Feature]:
        raise error

    monkeypatch.setattr(ChainA2320, "input_features", inputs)

    session = prepare_chain({ChainA2320, ChainB2320, ChainLeaf2320})

    result = result_for(session.resolution_report())
    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainA2320].stage == "inputs"
    assert str(error) in result.eliminations[ChainA2320].reason


@pytest.mark.parametrize("selection", ["sole", "class", "class-name"])
@pytest.mark.parametrize("marked", [False, True], ids=["ordinary-error", "match-abort"])
def test_explicit_or_sole_input_declaration_errors_keep_the_original_failure(
    selection: str, marked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ValueError("explicit input declaration_chain_2320")
    if marked:
        escalate_match_abort(error)

    def inputs(self: ChainStub2320, options: Options, name: FeatureName) -> set[Feature]:
        raise error

    monkeypatch.setattr(ChainA2320, "input_features", inputs)
    groups: set[type[FeatureGroup]] = {ChainA2320}
    scope: type[FeatureGroup] | str | None = None
    if selection != "sole":
        groups.update({ChainB2320, ChainLeaf2320})
        scope = ChainA2320 if selection == "class" else ChainA2320.__name__

    with pytest.raises(ValueError) as caught:
        prepare_chain(groups, feature=Feature(CHOICE, feature_group=scope))

    assert caught.value is error
