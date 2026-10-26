"""Input-chain attempts respect source scopes, preserve refusals and reuse per-column outcomes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, register_plugin
from mloda.core.prepare.identify_feature_group import FeatureResolutionError
from mloda.provider import FeatureGroup
from mloda.steward import ReaderProbeScope, reader_probe_scope
from mloda.user import DataAccessCollection, Feature, ParallelizationMode, PluginCollector
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from tests.test_core.test_prepare.input_chain_helpers import (
    CHOICE,
    LEAF,
    LOAD_CALLS,
    MATCH_CALLS,
    MISSING,
    PROBE_CALLS,
    RAW_A,
    RAW_B,
    SOURCE_KEY,
    ChainA2320,
    ChainB2320,
    ChainChecked2320,
    ChainGate2320,
    ChainLeaf2320,
    ChainOpen2320,
    ChainOtherChecked2320,
    chain_access,
    prepare_chain,
    result_for,
)


@pytest.fixture(autouse=True)
def _reset_source_counters() -> Iterator[None]:
    MATCH_CALLS.clear()
    PROBE_CALLS.clear()
    LOAD_CALLS.clear()
    yield
    MATCH_CALLS.clear()
    PROBE_CALLS.clear()
    LOAD_CALLS.clear()


@pytest.mark.parametrize("name", [CHOICE, f"{CHOICE}__literal"], ids=["plain", "literal-separator"])
@pytest.mark.parametrize("state", ["absent", "refused", "present"])
def test_derived_name_competes_with_a_real_csv_column(
    name: str, state: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ChainA2320, "MATCHED_NAMES", frozenset({name}))
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    csv_path = tmp_path / "chain2320.csv"
    contents = f"{name}\n11\n" if state == "absent" else f"{name},{RAW_A}\n11,5\n"
    csv_path.write_text(contents, encoding="utf-8")
    gate = ChainGate2320({RAW_A} if state == "refused" else (), CsvFG)

    session = prepare_chain(
        {ChainA2320, CsvFG},
        feature=Feature(name),
        access=DataAccessCollection(files={"chain_file_2320": str(csv_path)}),
        extender=gate,
    )

    result = result_for(session.resolution_report(), name)
    expected = ChainA2320 if state == "present" else CsvFG
    assert set(result.identified) == {expected}
    assert len(gate.calls_for(CsvFG, RAW_A)) == 1
    if state == "present":
        assert gate.calls_for(CsvFG, name) == []
        assert [record.feature_name for record in session.resolution_report()] == [name, RAW_A]
    else:
        assert len(gate.calls_for(CsvFG, name)) == 1
        assert result.eliminations[ChainA2320].stage == "inputs"
        assert RAW_A in result.eliminations[ChainA2320].reason
        values = session.run(parallelization_modes={ParallelizationMode.SYNC})
        assert values[0][name].to_pylist() == [11]
    expected_names = [(RAW_A,)] if state == "present" else [(RAW_A,), (name,)]
    assert sorted(context.feature_names for context in gate.contexts) == sorted(expected_names)
    assert len(gate.errors) == (1 if state == "refused" else 0)


def test_explicit_csv_scope_does_not_attempt_a_derived_candidates_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    csv_path = tmp_path / "explicit2320.csv"
    csv_path.write_text(f"{CHOICE},{RAW_A}\n11,5\n", encoding="utf-8")
    gate = ChainGate2320({RAW_A}, CsvFG)

    session = prepare_chain(
        {ChainA2320, CsvFG},
        feature=Feature(CHOICE, feature_group=CsvFG),
        access=DataAccessCollection(files={"chain_file_2320": str(csv_path)}),
        extender=gate,
    )

    assert set(result_for(session.resolution_report()).identified) == {CsvFG}
    assert gate.calls_for(CsvFG, RAW_A) == []
    assert MATCH_CALLS[ChainA2320, CHOICE] == 0


@pytest.mark.parametrize("state", ["absent", "refused"])
def test_pointed_csv_column_survives_an_unresolvable_derived_candidate(
    state: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    csv_path = tmp_path / "pointed2320.csv"
    csv_path.write_text(f"{CHOICE}\n11\n", encoding="utf-8")
    gate = ChainGate2320({RAW_A} if state == "refused" else (), CsvFG)

    session = prepare_chain(
        {ChainA2320, CsvFG}, feature=Feature(CHOICE, options={"CsvFG": str(csv_path)}), extender=gate
    )

    result = result_for(session.resolution_report())
    assert set(result.identified) == {CsvFG}
    assert result.eliminations[ChainA2320].stage == "inputs"
    values = session.run(parallelization_modes={ParallelizationMode.SYNC})
    assert values[0][CHOICE].to_pylist() == [11]


@pytest.mark.parametrize("state", ["absent", "refused", "present"])
def test_open_route_steps_aside_only_when_the_derived_chain_resolves(
    state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (Feature(RAW_A, forward_group=False),))
    columns = {CHOICE: [11]}
    gate = ChainGate2320({RAW_A} if state == "refused" else (), ChainChecked2320)

    session = prepare_chain(
        {ChainA2320, ChainOpen2320, ChainChecked2320},
        feature=Feature(CHOICE, options={ChainOpen2320.__name__: {SOURCE_KEY: columns}}),
        access=chain_access(**({RAW_A: [5]} if state != "absent" else {})),
        extender=gate,
    )

    result = result_for(session.resolution_report())
    expected = ChainA2320 if state == "present" else ChainOpen2320
    assert set(result.identified) == {expected}
    if state == "present":
        assert result.eliminations[ChainOpen2320].stage == "input_data"
    else:
        assert result.eliminations[ChainA2320].stage == "inputs"
        assert RAW_A in result.eliminations[ChainA2320].reason
        values = session.run(parallelization_modes={ParallelizationMode.SYNC})
        assert values[0][CHOICE].to_pylist() == [11]


def test_refused_input_loses_without_refusing_another_column_from_the_same_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (RAW_B,))
    gate = ChainGate2320({RAW_A}, ChainChecked2320)

    session = prepare_chain(
        {ChainA2320, ChainB2320, ChainChecked2320},
        access=chain_access(**{RAW_A: [1], RAW_B: [2]}),
        extender=gate,
    )

    result = result_for(session.resolution_report())
    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainA2320].stage == "inputs"
    assert "refused input" in result.eliminations[ChainA2320].reason
    assert len(gate.calls_for(ChainChecked2320, RAW_A)) == 1
    assert len(gate.calls_for(ChainChecked2320, RAW_B)) == 1
    assert PROBE_CALLS[ChainChecked2320, RAW_A] == 0
    assert PROBE_CALLS[ChainChecked2320, RAW_B] == 1
    assert LOAD_CALLS == []


def test_all_refused_chains_preserve_the_original_exception_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (RAW_B,))
    gate = ChainGate2320({RAW_A, RAW_B}, ChainChecked2320)

    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain(
            {ChainA2320, ChainB2320, ChainChecked2320},
            access=chain_access(**{RAW_A: [1], RAW_B: [2]}),
            extender=gate,
        )

    error = caught.value
    assert error.feature_name == CHOICE
    assert error.result.failure_kind == "none"
    assert len(gate.errors) == len(error.refusals) == 2
    assert {id(refusal) for refusal in error.refusals} == {id(refusal) for refusal in gate.errors}
    assert error.__cause__ is error.refusals[0]
    for candidate, name in ((ChainA2320, RAW_A), (ChainB2320, RAW_B)):
        elimination = error.result.eliminations[candidate]
        assert elimination.stage == "inputs"
        assert name in elimination.reason
        assert "refused input" in elimination.reason
        assert candidate.__name__ in str(error)
        assert len(gate.calls_for(ChainChecked2320, name)) == 1
        assert PROBE_CALLS[ChainChecked2320, name] == 0


def test_shared_refused_input_is_not_probed_again_for_the_next_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (RAW_A,))
    gate = ChainGate2320({RAW_A}, ChainChecked2320)

    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain({ChainA2320, ChainB2320, ChainChecked2320}, access=chain_access(**{RAW_A: [1]}), extender=gate)

    assert caught.value.result.failure_kind == "none"
    assert len(gate.calls_for(ChainChecked2320, RAW_A)) == 1
    assert len(gate.errors) == 1
    assert caught.value.refusals == (gate.errors[0],)
    assert caught.value.__cause__ is gate.errors[0]
    assert caught.value.result.eliminations[ChainA2320].stage == "inputs"
    assert caught.value.result.eliminations[ChainB2320].stage == "inputs"


def test_success_and_refusal_probe_caches_are_reset_for_the_next_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (RAW_B,))
    gate = ChainGate2320({RAW_A}, ChainChecked2320)
    access = chain_access(**{RAW_A: [1], RAW_B: [2]})

    for run_number in (1, 2):
        session = prepare_chain({ChainA2320, ChainB2320, ChainChecked2320}, access=access, extender=gate)
        assert set(result_for(session.resolution_report()).identified) == {ChainB2320}
        assert len(gate.calls_for(ChainChecked2320, RAW_A)) == run_number
        assert len(gate.calls_for(ChainChecked2320, RAW_B)) == run_number
        assert PROBE_CALLS[ChainChecked2320, RAW_B] == run_number


@pytest.mark.parametrize("strict", [False, True], ids=["not-enabled", "strict-dropped"])
def test_chain_attempts_do_not_probe_or_audit_inaccessible_loaded_readers(
    strict: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    gate = ChainGate2320({RAW_A}, ChainChecked2320)
    groups: set[type[FeatureGroup]] = {ChainA2320, ChainB2320, ChainLeaf2320}
    if strict:
        PluginRegistry.default().clear()
        for group in groups:
            register_plugin(group, replace=True)
        register_plugin(ChainGate2320, replace=True)
        collector = PluginCollector.enabled_feature_groups({*groups, ChainChecked2320}).set_strict_mode("strict")
    else:
        collector = PluginCollector.enabled_feature_groups(groups)

    session = prepare_chain(groups, collector=collector, access=chain_access(**{RAW_A: [1]}), extender=gate)

    result = result_for(session.resolution_report())
    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainA2320].stage == "inputs"
    assert PROBE_CALLS[ChainChecked2320, RAW_A] == 0
    assert gate.contexts == []
    assert gate.errors == []


def test_chain_attempts_without_an_extender_do_not_inherit_an_outer_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (RAW_A,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))
    outer_gate = ChainGate2320({RAW_A}, ChainChecked2320)

    with reader_probe_scope(ReaderProbeScope(outer_gate, None, None, None, None)):
        session = prepare_chain({ChainA2320, ChainB2320, ChainChecked2320}, access=chain_access(**{RAW_A: [1]}))

    assert set(result_for(session.resolution_report()).identified) == {ChainA2320}
    assert outer_gate.contexts == []
    assert PROBE_CALLS[ChainChecked2320, RAW_A] == 1


def test_input_pointer_excludes_other_loaded_readers_before_any_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    pointed = {ChainChecked2320.__name__: {SOURCE_KEY: {RAW_A: [1]}}}
    monkeypatch.setattr(ChainA2320, "INPUTS", (Feature(RAW_A, options=pointed),))
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))
    gate = ChainGate2320({RAW_A}, ChainOtherChecked2320)

    session = prepare_chain({ChainA2320, ChainB2320, ChainChecked2320, ChainOtherChecked2320}, extender=gate)

    assert set(result_for(session.resolution_report()).identified) == {ChainA2320}
    assert gate.calls_for(ChainOtherChecked2320, RAW_A) == []
    assert PROBE_CALLS[ChainOtherChecked2320, RAW_A] == 0
    assert len(gate.calls_for(ChainChecked2320, RAW_A)) == 1
    assert PROBE_CALLS[ChainChecked2320, RAW_A] == 1


def test_cycle_does_not_hide_a_refused_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (CHOICE, RAW_A))
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
    assert len(gate.calls_for(ChainChecked2320, RAW_A)) == 1
    assert [record.feature_name for record in session.resolution_report()] == [CHOICE, LEAF]
