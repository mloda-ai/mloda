"""Candidate selection checks complete input chains without changing explicit or sole choices."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import (
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    resolve_or_raise,
)
from mloda.core.prepare.resolution_types import EvaluationResult, ResolutionRecord
from mloda.provider import FeatureGroup
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda, resolve_feature
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_prepare.input_chain_helpers import (
    CHOICE,
    LEAF,
    MATCH_CALLS,
    MIDDLE,
    MISSING,
    OTHER_MISSING,
    ChainA2320,
    ChainB2320,
    ChainBChild2320,
    ChainC2320,
    ChainLeaf2320,
    ChainMiddle2320,
    ChainStub2320,
    prepare_chain,
    result_for,
)


GROUPS: set[type[FeatureGroup]] = {ChainA2320, ChainB2320, ChainMiddle2320, ChainLeaf2320}


@pytest.fixture(autouse=True)
def _reset_matches() -> Iterator[None]:
    MATCH_CALLS.clear()
    yield
    MATCH_CALLS.clear()


def _evaluate(*groups: type[FeatureGroup], feature: Feature | None = None) -> EvaluationResult:
    return IdentifyFeatureGroupClass.evaluate(
        feature if feature is not None else Feature(CHOICE), {group: {PyArrowTable} for group in groups}, None, None
    )


def test_two_hop_failure_selects_the_candidate_whose_whole_chain_resolves() -> None:
    session = prepare_chain(GROUPS)

    records = session.resolution_report()
    result = result_for(records)
    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainA2320].stage == "inputs"
    assert MIDDLE in result.eliminations[ChainA2320].reason
    assert MISSING in result.eliminations[ChainA2320].reason
    assert [(record.feature_name, record.requested) for record in records] == [(CHOICE, True), (LEAF, False)]
    assert session.engine is not None
    assert set(session.engine.feature_group_collection) == {ChainB2320, ChainLeaf2320}
    assert {
        str(feature.name) for features in session.engine.feature_group_collection.values() for feature in features
    } == {CHOICE, LEAF}


def test_evaluate_uses_the_same_chain_feasibility() -> None:
    result = _evaluate(*GROUPS)

    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainA2320].stage == "inputs"


def test_diagnose_and_resolve_feature_agree_with_prepare() -> None:
    collector = PluginCollector.enabled_feature_groups(GROUPS)
    diagnosis = mloda.diagnose([CHOICE], compute_frameworks=[PyArrowTable], plugin_collector=collector)
    resolved = resolve_feature(CHOICE, compute_frameworks={PyArrowTable}, plugin_collector=collector)

    assert diagnosis.complete is True
    assert set(result_for(diagnosis.records).identified) == {ChainB2320}
    assert resolved.error is None
    assert resolved.feature_group is ChainB2320


def test_all_failed_chains_report_each_candidate_and_failing_input() -> None:
    groups = {ChainA2320, ChainC2320, ChainMiddle2320}
    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain(groups)

    error = caught.value
    assert error.feature_name == CHOICE
    assert error.result.failure_kind == "none"
    assert error.partial_records == ()
    for candidate, missing in ((ChainA2320, MISSING), (ChainC2320, OTHER_MISSING)):
        elimination = error.result.eliminations[candidate]
        assert elimination.stage == "inputs"
        assert missing in elimination.reason
        assert candidate.__name__ in str(error)
        assert missing in str(error)
    assert error.refusals == ()
    assert error.__cause__ is None


def test_all_failed_diagnosis_reports_the_requested_name_and_inputs_stage() -> None:
    diagnosis = mloda.diagnose(
        [CHOICE],
        compute_frameworks=[PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups({ChainA2320, ChainC2320, ChainMiddle2320}),
    )

    assert diagnosis.complete is False
    assert diagnosis.feature_name == CHOICE
    assert diagnosis.records == []
    assert diagnosis.failed_result is not None
    assert diagnosis.failed_result.failure_kind == "none"
    assert diagnosis.failed_result.eliminations[ChainA2320].stage == "inputs"
    assert diagnosis.failed_result.eliminations[ChainC2320].stage == "inputs"


def test_multiple_viable_chains_keep_the_existing_ambiguity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (LEAF,))

    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain(GROUPS)

    assert caught.value.result.failure_kind == "multiple"
    assert set(caught.value.result.identified) == {ChainA2320, ChainB2320}
    assert caught.value.feature_name == CHOICE


def test_viable_subclass_keeps_precedence_among_viable_candidates() -> None:
    result = _evaluate(ChainB2320, ChainBChild2320, ChainLeaf2320)

    assert set(result.identified) == {ChainBChild2320}
    assert result.specialized_from == (ChainB2320,)


def test_unresolvable_subclass_does_not_hide_a_viable_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainBChild2320, "INPUTS", (MISSING,))

    session = prepare_chain({ChainB2320, ChainBChild2320, ChainLeaf2320})

    result = result_for(session.resolution_report())
    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainBChild2320].stage == "inputs"


def test_a_sole_candidate_still_fails_at_the_missing_dependency() -> None:
    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain({ChainA2320, ChainMiddle2320})

    assert caught.value.feature_name == MISSING
    assert [record.feature_name for record in caught.value.partial_records] == [CHOICE, MIDDLE]
    assert all(record.result.failure_kind is None for record in caught.value.partial_records)


@pytest.mark.parametrize("as_string", [False, True], ids=["class", "class-name"])
def test_explicit_choice_is_narrowed_before_inputs_and_keeps_its_failure(as_string: bool) -> None:
    scope = ChainA2320.__name__ if as_string else ChainA2320

    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain(GROUPS, feature=Feature(CHOICE, feature_group=scope))

    assert caught.value.feature_name == MISSING
    assert set(caught.value.partial_records[0].result.identified) == {ChainA2320}
    assert MATCH_CALLS[ChainB2320, CHOICE] == 0


@pytest.mark.parametrize("root_form", ["none", "unimplemented", "empty"])
def test_all_root_protocol_forms_are_viable_candidates(root_form: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def inputs(self: ChainStub2320, options: Options, name: FeatureName) -> set[Feature] | None:
        if root_form == "unimplemented":
            raise NotImplementedError
        return set() if root_form == "empty" else None

    monkeypatch.setattr(ChainB2320, "input_features", inputs)

    result = _evaluate(ChainA2320, ChainB2320, ChainMiddle2320)

    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainA2320].stage == "inputs"


@pytest.mark.parametrize("two_hop", [False, True], ids=["self-cycle", "two-node-cycle"])
def test_cycles_reach_the_existing_graph_cycle_error(two_hop: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (MIDDLE if two_hop else CHOICE,))
    monkeypatch.setattr(ChainMiddle2320, "INPUTS", (CHOICE,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))

    with pytest.raises(ValueError, match="cycle detected in feature graph") as caught:
        prepare_chain(GROUPS)

    assert not isinstance(caught.value, FeatureResolutionError)


@pytest.mark.parametrize("two_hop", [False, True], ids=["self-cycle", "two-node-cycle"])
def test_cycle_back_edge_does_not_hide_a_second_missing_input(two_hop: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (MIDDLE if two_hop else CHOICE, MISSING))
    monkeypatch.setattr(ChainMiddle2320, "INPUTS", (CHOICE,))

    session = prepare_chain(GROUPS)

    result = result_for(session.resolution_report())
    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainA2320].stage == "inputs"
    assert MISSING in result.eliminations[ChainA2320].reason


def test_winners_matched_options_are_replayed_without_loser_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "WRITES", {"loser_chain_2320": "not-selected"})
    monkeypatch.setattr(ChainB2320, "WRITES", {"winner_chain_2320": "selected"})
    feature = Feature(CHOICE)

    session = prepare_chain(GROUPS, feature=feature)

    assert set(result_for(session.resolution_report()).identified) == {ChainB2320}
    assert session.engine is not None
    (planned,) = session.engine.feature_group_collection[ChainB2320]
    assert planned.options.get("winner_chain_2320") == "selected"
    assert planned.options.get("loser_chain_2320") is None
    assert planned.options.is_own("winner_chain_2320") is False
    assert feature.options.group == {}


def test_loser_preflight_does_not_mutate_its_provider_owned_input(monkeypatch: pytest.MonkeyPatch) -> None:
    declared_input = Feature(MISSING)

    def inputs(self: ChainStub2320, options: Options, name: FeatureName) -> set[Feature]:
        return {declared_input}

    monkeypatch.setattr(ChainA2320, "input_features", inputs)
    feature = Feature(CHOICE, options={"forwarded_chain_2320": 7})

    prepare_chain(GROUPS, feature=feature)

    assert declared_input.options.group == {}
    assert declared_input.child_options is None
    assert declared_input.resolving_consumer is None
    assert declared_input.resolving_path == ()


def test_shared_dependency_matching_is_reused_through_final_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (LEAF, MISSING))
    monkeypatch.setattr(ChainB2320, "INPUTS", (MIDDLE,))
    monkeypatch.setattr(ChainMiddle2320, "INPUTS", (LEAF,))

    session = prepare_chain(GROUPS)

    assert set(result_for(session.resolution_report()).identified) == {ChainB2320}
    assert MATCH_CALLS[ChainLeaf2320, LEAF] == 1
    assert MATCH_CALLS[ChainMiddle2320, MIDDLE] == 1
    assert [record.feature_name for record in session.resolution_report()] == [CHOICE, MIDDLE, LEAF]


def test_next_run_rechecks_a_previously_failed_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _evaluate(*GROUPS)
    assert set(first.identified) == {ChainB2320}
    monkeypatch.setattr(ChainMiddle2320, "INPUTS", (LEAF,))

    second = _evaluate(*GROUPS)

    assert second.failure_kind == "multiple"
    assert set(second.identified) == {ChainA2320, ChainB2320}


def test_failed_chain_cache_does_not_capture_outer_diagnostic_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))
    plugins: FeatureGroupEnvironmentMapping = {
        group: {PyArrowTable} for group in (ChainA2320, ChainB2320, ChainMiddle2320)
    }
    earlier_result = EvaluationResult(identified={ChainLeaf2320: {PyArrowTable}})
    earlier_records = [
        ResolutionRecord("first_requested_chain_2320", True, earlier_result),
        ResolutionRecord("second_requested_chain_2320", True, earlier_result),
    ]
    failures: list[EvaluationResult] = []

    with run_match_cache():
        for earlier in earlier_records:
            with pytest.raises(FeatureResolutionError) as caught:
                resolve_or_raise(Feature(CHOICE), plugins, partial_records=[earlier])
            assert caught.value.partial_records == (earlier,)
            failures.append(caught.value.result)

    first, second = failures
    assert first.failure_kind == second.failure_kind == "none"
    assert first.eliminations == second.eliminations
    assert first.eliminations[ChainA2320].stage == "inputs"
