"""Candidate selection resolves each candidate's input chain under the run's scopes (#2320)."""

from __future__ import annotations

import traceback
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError, IdentifyFeatureGroupClass
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import FeatureGroup, escalate_match_abort
from mloda.user import (
    DataAccessCollection,
    Feature,
    Index,
    JoinSpec,
    Link,
    FeatureName,
    Options,
    ParallelizationMode,
    PluginCollector,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
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
    ChainStub2320,
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


BAD_INPUT = "bad_input_chain_2320"


def _raise_in_plugin_2320(error: Exception) -> None:
    raise error


class ChainSub2320(ChainB2320):
    INPUTS = (MISSING,)


class ChainBadMatcher2320(ChainStub2320):
    """Matches only BAD_INPUT, and raises a marked abort when asked to."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) == BAD_INPUT:
            raise escalate_match_abort(ValueError("matcher abort chain_2320"))
        return False


class ChainRaisingExtender2320(Extender):
    """Raises on the FEATURE_GROUP_MATCHED hook for one feature name."""

    def __init__(self, name: str) -> None:
        self.priority = 100
        self.name = "raising_extender_2320"
        self.target = name

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_MATCHED}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        assert context is not None
        if context.feature_names == (self.target,):
            raise RuntimeError("extender failure chain_2320")
        return func(*args, **kwargs)


@pytest.mark.parametrize("with_source", [False, True], ids=["plain", "with-searched-source"])
@pytest.mark.parametrize("where", ["input_features", "set_feature_name"])
def test_a_sole_candidates_own_traceback_survives(
    where: str, with_source: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = KeyError("plugin failure chain_2320")

    def raising_frame_2320(*args: Any, **kwargs: Any) -> Any:
        _raise_in_plugin_2320(error)

    monkeypatch.setattr(ChainA2320, where, raising_frame_2320)
    groups: set[type[FeatureGroup]] = {ChainA2320}
    if with_source:
        groups.add(ChainChecked2320)

    with pytest.raises(KeyError) as caught:
        prepare_chain(groups, access=chain_access(**{RAW_A: [1]}) if with_source else None)

    assert caught.value is error
    assert "_raise_in_plugin_2320" in "".join(traceback.format_exception(caught.value))


def _failing_own_inputs(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def inputs(self: Any, options: Options, name: FeatureName) -> set[Feature]:
        raise error

    monkeypatch.setattr(ChainA2320, "input_features", inputs)


def _connection_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChainA2320, "INPUTS", (Feature(MISSING, connection=object()),))


@pytest.mark.parametrize("case", ["own-abort", "input-matcher-abort", "connection", "match-extender"])
@pytest.mark.parametrize("with_source", [False, True], ids=["plain", "source-has-choice"])
def test_framework_errors_in_a_failing_chain_propagate(
    case: str, with_source: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    groups: set[type[FeatureGroup]] = {ChainA2320, ChainB2320, ChainLeaf2320}
    if with_source:
        groups.add(ChainChecked2320)
    extender: Extender | None = None
    expected: Exception | None = None
    if case == "own-abort":
        expected = escalate_match_abort(ValueError("own abort chain_2320"))
        _failing_own_inputs(monkeypatch, expected)
    elif case == "input-matcher-abort":
        monkeypatch.setattr(ChainA2320, "INPUTS", (BAD_INPUT,))
        groups.add(ChainBadMatcher2320)
    elif case == "connection":
        _connection_input(monkeypatch)
    else:
        groups.add(ChainMiddle2320)
        extender = ChainRaisingExtender2320(MISSING)

    with pytest.raises((ValueError, RuntimeError)) as caught:
        prepare_chain(groups, extender=extender, access=chain_access(**{CHOICE: [1]}) if with_source else None)

    assert not isinstance(caught.value, FeatureResolutionError)
    if expected is not None:
        assert caught.value is expected


def test_an_unmarked_plain_error_in_a_failing_chain_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    _failing_own_inputs(monkeypatch, KeyError("plain chain_2320"))

    result = result_for(prepare_chain({ChainA2320, ChainB2320, ChainLeaf2320}).resolution_report())

    assert set(result.identified) == {ChainB2320}
    assert result.eliminations[ChainA2320].stage == "inputs"


@pytest.mark.parametrize("with_open_claim", [False, True], ids=["plain", "open-claim"])
def test_subclass_takeover_is_final_and_narrows_before_chain_checks(
    with_open_claim: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ChainB2320, "INPUTS", (RAW_A,))
    groups: set[type[FeatureGroup]] = {ChainB2320, ChainSub2320, ChainChecked2320}
    access = chain_access(**{RAW_A: [1]})
    if with_open_claim:
        monkeypatch.setattr(ChainSub2320, "INPUTS", (Feature(MISSING, forward_group=False),))
        groups.add(ChainOpen2320)
        feature = Feature(CHOICE, options={ChainOpen2320.__name__: {SOURCE_KEY: {CHOICE: [11]}}})
        result = result_for(prepare_chain(groups, feature=feature, access=access).resolution_report())
        assert set(result.identified) == {ChainOpen2320}
        assert result.eliminations[ChainSub2320].stage == "inputs"
    else:
        with pytest.raises(FeatureResolutionError) as caught:
            prepare_chain(groups, access=access)
        assert caught.value.feature_name == MISSING
    assert PROBE_CALLS[ChainChecked2320, RAW_A] == 0


@pytest.mark.parametrize("varies", ["plugins", "access", "links"])
def test_cached_chain_outcomes_do_not_cross_runs(varies: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if varies == "links":
        monkeypatch.setattr(ChainA2320, "INPUTS", (LEAF,))
        monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))
        monkeypatch.setattr(ChainLeaf2320, "index_columns", classmethod(lambda cls: [Index(("id_chain_2320",))]))
        good = Link("inner", JoinSpec(ChainLeaf2320, "id_chain_2320"), JoinSpec(ChainB2320, "id_chain_2320"))
        bad = Link("inner", JoinSpec(ChainLeaf2320, "wrong_chain_2320"), JoinSpec(ChainB2320, "wrong_chain_2320"))
        plugins: FeatureGroupEnvironmentMapping = {
            group: {PyArrowTable} for group in (ChainA2320, ChainB2320, ChainLeaf2320)
        }
        with run_match_cache():
            resolved = IdentifyFeatureGroupClass.evaluate(Feature(CHOICE), plugins, {good}, None)
            failed = IdentifyFeatureGroupClass.evaluate(Feature(CHOICE), plugins, {bad}, None)
        assert set(resolved.identified) == {ChainA2320}
        assert failed.eliminations[ChainA2320].stage == "inputs"
        return
    monkeypatch.setattr(ChainA2320, "INPUTS", (LEAF,) if varies == "plugins" else (RAW_A,))
    monkeypatch.setattr(ChainB2320, "INPUTS", (MISSING,))
    base: set[type[FeatureGroup]] = {ChainA2320, ChainB2320}
    full: set[type[FeatureGroup]] = {*base, ChainLeaf2320} if varies == "plugins" else {*base, ChainChecked2320}
    first_access = chain_access(**{RAW_A: [1]}) if varies == "access" else None
    second_access = chain_access() if varies == "access" else None

    first = result_for(prepare_chain(full, access=first_access).resolution_report())
    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain(base if varies == "plugins" else full, access=second_access)

    assert set(first.identified) == {ChainA2320}
    assert caught.value.result.eliminations[ChainA2320].stage == "inputs"


def test_failure_text_stays_small_when_every_level_has_two_failing_candidates() -> None:
    depth = 6
    groups: list[type[FeatureGroup]] = []
    for level in range(depth):
        for side in "AB":
            attrs = {
                "MATCHED_NAMES": frozenset({f"level{level}_chain_2320"}),
                "INPUTS": (f"level{level + 1}_chain_2320",),
            }
            groups.append(type(f"Level{level}{side}2320", (ChainStub2320,), attrs))

    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain(set(groups), feature=Feature("level0_chain_2320"))

    assert len(str(caught.value)) < 6000
    for group in groups[:2]:
        elimination = caught.value.result.eliminations[group]
        assert elimination.stage == "inputs"
        assert "level1_chain_2320" in elimination.reason


class ChainUnsupported2320(ChainA2320):
    @classmethod
    def supports_compute_framework(cls, *args: Any, **kwargs: Any) -> bool:
        return False


class ChainOtherFramework2320(ChainA2320):
    FRAMEWORK_RULE = {PythonDictFramework}


@pytest.mark.parametrize("derived", [ChainUnsupported2320, ChainOtherFramework2320], ids=["capability", "framework"])
def test_open_route_step_aside_does_not_probe_a_candidate_eliminated_later(
    derived: type[FeatureGroup], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(derived, "INPUTS", (Feature(RAW_A, forward_group=False),))
    gate = ChainGate2320(reader=ChainChecked2320)

    with pytest.raises(FeatureResolutionError) as caught:
        prepare_chain(
            {derived, ChainOpen2320, ChainChecked2320},
            feature=Feature(CHOICE, options={ChainOpen2320.__name__: {SOURCE_KEY: {CHOICE: [11]}}}),
            access=chain_access(**{RAW_A: [5]}),
            extender=gate,
        )

    assert caught.value.result.eliminations[derived].stage != "inputs"
    assert gate.calls_for(ChainChecked2320, RAW_A) == []
    assert PROBE_CALLS[ChainChecked2320, RAW_A] == 0


class ChainCountingExtender2320(Extender):
    """Counts FEATURE_GROUP_MATCHED hook calls per feature name."""

    def __init__(self) -> None:
        self.priority = 100
        self.name = "counting_extender_2320"
        self.counts: Counter[str] = Counter()

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_MATCHED}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        assert context is not None
        self.counts.update(context.feature_names)
        return func(*args, **kwargs)


def _linear_chain(depth: int) -> list[type[FeatureGroup]]:
    groups: list[type[FeatureGroup]] = []
    for level in range(depth):
        attrs: dict[str, Any] = {"MATCHED_NAMES": frozenset({f"level{level}_chain_2320"})}
        if level + 1 < depth:
            attrs["INPUTS"] = (f"level{level + 1}_chain_2320",)
        groups.append(type(f"Level{level}2320", (ChainStub2320,), attrs))
    return groups


@pytest.mark.parametrize("depth", [3, 150])
@pytest.mark.parametrize("with_access", [False, True], ids=["no-access", "access-without-chain-names"])
def test_a_single_candidate_chain_fires_the_match_hook_once_per_feature(depth: int, with_access: bool) -> None:
    groups = _linear_chain(depth)
    extender = ChainCountingExtender2320()

    prepare_chain(
        {*groups, ChainChecked2320},
        feature=Feature("level0_chain_2320"),
        access=chain_access(**{RAW_A: [1]}) if with_access else None,
        extender=extender,
    )

    assert extender.counts == {f"level{level}_chain_2320": 1 for level in range(depth)}


def test_a_derived_chain_steps_aside_a_source_that_has_the_root_name() -> None:
    groups = _linear_chain(3)

    session = prepare_chain(
        {*groups, ChainChecked2320},
        feature=Feature("level0_chain_2320"),
        access=chain_access(level0_chain_2320=[1]),
    )

    result = result_for(session.resolution_report(), "level0_chain_2320")
    assert set(result.identified) == {groups[0]}
