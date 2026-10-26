"""Candidate selection resolves each candidate's input chain under the run's scopes (#2320)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mloda.core.prepare.identify_feature_group import FeatureResolutionError
from mloda.provider import FeatureGroup
from mloda.user import DataAccessCollection, Feature, ParallelizationMode, PluginCollector
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from tests.test_core.test_prepare.input_chain_helpers import (
    CHOICE,
    LEAF,
    MATCH_CALLS,
    MIDDLE,
    MISSING,
    OTHER_MISSING,
    PROBE_CALLS,
    RAW_A,
    RAW_B,
    SOURCE_KEY,
    ChainA2320,
    ChainB2320,
    ChainC2320,
    ChainChecked2320,
    ChainGate2320,
    ChainLeaf2320,
    ChainMiddle2320,
    ChainOpen2320,
    chain_access,
    prepare_chain,
    result_for,
)


GROUPS: set[type[FeatureGroup]] = {ChainA2320, ChainB2320, ChainMiddle2320, ChainLeaf2320}


@pytest.fixture(autouse=True)
def _reset_counters() -> Iterator[None]:
    MATCH_CALLS.clear()
    PROBE_CALLS.clear()
    yield
    MATCH_CALLS.clear()
    PROBE_CALLS.clear()


def test_two_hop_failure_selects_the_candidate_whose_whole_chain_resolves() -> None:
    result = result_for(prepare_chain(GROUPS).resolution_report())

    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainA2320].stage == "inputs"
    assert MIDDLE in result.eliminations[ChainA2320].reason
    assert MISSING in result.eliminations[ChainA2320].reason


def test_a_sole_candidate_still_fails_at_the_missing_dependency() -> None:
    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain({ChainA2320, ChainMiddle2320})

    assert caught.value.feature_name == MISSING


def test_all_failed_chains_report_each_candidate_and_failing_input() -> None:
    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain({ChainA2320, ChainC2320, ChainMiddle2320})

    error = caught.value
    assert error.result.failure_kind == "none"
    for candidate, missing in ((ChainA2320, MISSING), (ChainC2320, OTHER_MISSING)):
        assert error.result.eliminations[candidate].stage == "inputs"
        assert missing in error.result.eliminations[candidate].reason
        assert candidate.__name__ in str(error)


def test_all_refused_chains_carry_the_original_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (RAW_B,))
    gate = ChainGate2320({RAW_A, RAW_B}, ChainChecked2320)

    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain(
            {ChainA2320, ChainB2320, ChainChecked2320}, access=chain_access(**{RAW_A: [1], RAW_B: [2]}), extender=gate
        )

    error = caught.value
    assert len(gate.errors) == len(error.refusals) == 2
    assert {id(refusal) for refusal in error.refusals} == {id(refusal) for refusal in gate.errors}
    assert error.__cause__ is error.refusals[0]
    for candidate in (ChainA2320, ChainB2320):
        assert "refused input" in error.result.eliminations[candidate].reason


def test_refused_chain_loses_and_a_shared_source_is_probed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (RAW_A,))
    gate = ChainGate2320({RAW_A}, ChainChecked2320)

    with pytest.raises(FeatureResolutionError):
        prepare_chain({ChainA2320, ChainB2320, ChainChecked2320}, access=chain_access(**{RAW_A: [1]}), extender=gate)

    assert len(gate.calls_for(ChainChecked2320, RAW_A)) == 1


def test_chain_attempts_do_not_probe_inaccessible_loaded_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    gate = ChainGate2320({RAW_A}, ChainChecked2320)
    groups: set[type[FeatureGroup]] = {ChainA2320, ChainB2320, ChainLeaf2320}

    session = prepare_chain(
        groups,
        collector=PluginCollector.enabled_feature_groups(groups),
        access=chain_access(**{RAW_A: [1]}),
        extender=gate,
    )

    result = result_for(session.resolution_report())
    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainA2320].stage == "inputs"
    assert PROBE_CALLS[ChainChecked2320, RAW_A] == 0
    assert gate.contexts == []


def test_multiple_viable_chains_keep_the_existing_ambiguity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (LEAF,))

    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain(GROUPS)

    assert caught.value.result.failure_kind == "multiple"
    assert set(caught.value.result.identified) == {ChainA2320, ChainB2320}


def test_explicit_choice_is_narrowed_before_inputs_and_keeps_its_failure() -> None:
    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain(GROUPS, feature=Feature(CHOICE, feature_group=ChainA2320))

    assert caught.value.feature_name == MISSING
    assert MATCH_CALLS[ChainB2320, CHOICE] == 0


@pytest.mark.parametrize("two_hop", [False, True], ids=["self-cycle", "two-node-cycle"])
def test_cycles_reach_the_existing_graph_cycle_error(two_hop: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (MIDDLE if two_hop else CHOICE,))
    monkeypatch.setattr(ChainMiddle2320, "INPUTS", (CHOICE,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))

    with pytest.raises(ValueError, match="cycle detected in feature graph") as caught:
        prepare_chain(GROUPS)

    assert not isinstance(caught.value, FeatureResolutionError)


@pytest.mark.parametrize("state", ["absent", "refused", "present"])
def test_derived_name_competes_with_a_real_csv_column(
    state: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    csv_path = tmp_path / "chain2320.csv"
    csv_path.write_text(f"{CHOICE}\n11\n" if state == "absent" else f"{CHOICE},{RAW_A}\n11,5\n", encoding="utf-8")
    gate = ChainGate2320({RAW_A} if state == "refused" else (), CsvFG)

    session = prepare_chain(
        {ChainA2320, CsvFG},
        access=DataAccessCollection(files={"chain_file_2320": str(csv_path)}),
        extender=gate,
    )

    result = result_for(session.resolution_report())
    assert set(result.identified) == {ChainA2320 if state == "present" else CsvFG}
    if state != "present":
        assert result.eliminations[ChainA2320].stage == "inputs"
        assert session.run(parallelization_modes={ParallelizationMode.SYNC})[0][CHOICE].to_pylist() == [11]


@pytest.mark.parametrize("state", ["absent", "refused", "present"])
def test_open_route_steps_aside_only_when_the_derived_chain_resolves(
    state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (Feature(RAW_A, forward_group=False),))
    gate = ChainGate2320({RAW_A} if state == "refused" else (), ChainChecked2320)

    session = prepare_chain(
        {ChainA2320, ChainOpen2320, ChainChecked2320},
        feature=Feature(CHOICE, options={ChainOpen2320.__name__: {SOURCE_KEY: {CHOICE: [11]}}}),
        access=chain_access(**({RAW_A: [5]} if state != "absent" else {})),
        extender=gate,
    )

    result = result_for(session.resolution_report())
    assert set(result.identified) == {ChainA2320 if state == "present" else ChainOpen2320}
    if state != "present":
        assert result.eliminations[ChainA2320].stage == "inputs"
