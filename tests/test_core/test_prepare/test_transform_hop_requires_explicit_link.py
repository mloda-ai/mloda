"""A FeatureGroupStep whose parents live on different, unlinked source-framework instances
must raise a "missing Links" ValueError at plan-build time, not silently bind only one hop's data.
"""

from pathlib import Path
from typing import Any, ClassVar

import pytest

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.helpers.probe_runner import run_probes
from tests.test_core.test_prepare import link_side_paths_probe as hd_probe


class UnlinkedRootA(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"unlinked_root_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"unlinked_root_a": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class UnlinkedRootB(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"unlinked_root_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"unlinked_root_b": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class UnlinkedConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("unlinked_root_a"), Feature("unlinked_root_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if "unlinked_root_a" in data.column_names:
            return data.append_column("unlinked_consumer_result", data["unlinked_root_a"])
        if "unlinked_root_b" in data.column_names:
            return data.append_column("unlinked_consumer_result", data["unlinked_root_b"])
        raise ValueError(f"neither present: {data.column_names}")

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"unlinked_consumer_result"}


def test_two_unlinked_source_framework_instances_raise_missing_links_error_at_prepare_time() -> None:
    with pytest.raises(ValueError) as exc_info:
        mloda.prepare(
            features=[Feature("unlinked_consumer_result")],
            links=set(),
            compute_frameworks=[PandasDataFrame, PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({UnlinkedRootA, UnlinkedRootB, UnlinkedConsumer}),
        )

    error_message = str(exc_info.value)

    assert "Link" in error_message, "Error should mention Links as the missing piece of guidance"
    assert "UnlinkedRootA" in error_message, "Error should name the first conflicting upstream feature group"
    assert "UnlinkedRootB" in error_message, "Error should name the second conflicting upstream feature group"


# A consumer needing a plain hop from a case-override subclass (same family as
# a join's own source side) plus that join's own destination side is fully covered by the single
# declared Link, so it must plan and run under every PYTHONHASHSEED, not just some of them.
_SUBCLASS_LINKED_HOP_AND_JOIN_PROBE = Path(__file__).with_name("subclass_linked_hop_and_join_probe.py")
_SUBCLASS_LINKED_HOP_AND_JOIN_SEEDS = [0, 1, 3, 4, 6]
_SUBCLASS_LINKED_HOP_AND_JOIN_EXPECTED = {"outcome": "accepted", "subclass_linked_x": "[111, 222, 333]"}


@pytest.mark.timeout(60)
def test_subclass_linked_plain_hop_and_join_plan_the_same_way_under_every_hash_seed() -> None:
    outputs = run_probes(
        _SUBCLASS_LINKED_HOP_AND_JOIN_PROBE,
        len(_SUBCLASS_LINKED_HOP_AND_JOIN_SEEDS),
        seeds=_SUBCLASS_LINKED_HOP_AND_JOIN_SEEDS,
    )

    assert len(outputs) == len(_SUBCLASS_LINKED_HOP_AND_JOIN_SEEDS)
    for seed, output in zip(_SUBCLASS_LINKED_HOP_AND_JOIN_SEEDS, outputs):
        assert output == _SUBCLASS_LINKED_HOP_AND_JOIN_EXPECTED, (
            f"PYTHONHASHSEED={seed} produced {output}, expected {_SUBCLASS_LINKED_HOP_AND_JOIN_EXPECTED}"
        )


# A consumer reads two plain hops out of one PythonDictFramework source
# (p3_a via P3GateA, p3_b via P3GateB) whose from_feature_group classes, P3A and P3B(P3A), are
# related only by subclassing for code reuse, not by sharing any parent feature/uuid. Unlike the
# case-override case above, neither hop's parent is an ancestor of the other's, and no Link ties
# the two gates together, so `_entries_linked`'s bare `issubclass` check wrongly merges the two
# hops into one binding and drops one column's data instead of keeping both.
_SUBCLASS_SIBLING_PLAIN_HOPS_PROBE = Path(__file__).with_name("subclass_sibling_plain_hops_probe.py")
_SUBCLASS_SIBLING_PLAIN_HOPS_SEEDS = [0, 1, 3, 4, 6]
_SUBCLASS_SIBLING_PLAIN_HOPS_EXPECTED = {"outcome": "accepted", "p3_x": "[6, 8, 10]"}


@pytest.mark.timeout(60)
def test_subclass_sibling_plain_hops_both_survive_under_every_hash_seed() -> None:
    outputs = run_probes(
        _SUBCLASS_SIBLING_PLAIN_HOPS_PROBE,
        len(_SUBCLASS_SIBLING_PLAIN_HOPS_SEEDS),
        seeds=_SUBCLASS_SIBLING_PLAIN_HOPS_SEEDS,
    )

    assert len(outputs) == len(_SUBCLASS_SIBLING_PLAIN_HOPS_SEEDS)
    for seed, output in zip(_SUBCLASS_SIBLING_PLAIN_HOPS_SEEDS, outputs):
        assert output == _SUBCLASS_SIBLING_PLAIN_HOPS_EXPECTED, (
            f"PYTHONHASHSEED={seed} produced {output}, expected {_SUBCLASS_SIBLING_PLAIN_HOPS_EXPECTED}"
        )


# ScRootA and ScRootB(ScRootA) are related only by subclassing for code reuse: each is its own
# independent DataCreator root with zero shared physical lineage. Since the two parents share no
# ancestor, the plan must reject this shape deterministically at plan-build time with a "missing
# Links" ValueError under every hash seed, for both the cross-framework and same-framework variants,
# never a crash (from the candidate loop in `prepare_tfs_right_cfw` / `_drop_tfs_source_if_possible`
# resolving to the wrong sibling's physical instance) and never a silently-wrong result.
_SUBCLASS_UNRELATED_ROOTS_CROSS_FRAMEWORK_PROBE = Path(__file__).with_name(
    "subclass_unrelated_roots_cross_framework_probe.py"
)
_SUBCLASS_UNRELATED_ROOTS_SAME_FRAMEWORK_PROBE = Path(__file__).with_name(
    "subclass_unrelated_roots_same_framework_probe.py"
)
_SUBCLASS_UNRELATED_ROOTS_SEEDS = [0, 1, 3, 4, 6]


@pytest.mark.timeout(60)
def test_subclass_unrelated_roots_reject_missing_links_under_every_hash_seed() -> None:
    cross_outputs = run_probes(
        _SUBCLASS_UNRELATED_ROOTS_CROSS_FRAMEWORK_PROBE,
        len(_SUBCLASS_UNRELATED_ROOTS_SEEDS),
        seeds=_SUBCLASS_UNRELATED_ROOTS_SEEDS,
    )
    same_outputs = run_probes(
        _SUBCLASS_UNRELATED_ROOTS_SAME_FRAMEWORK_PROBE,
        len(_SUBCLASS_UNRELATED_ROOTS_SEEDS),
        seeds=_SUBCLASS_UNRELATED_ROOTS_SEEDS,
    )

    assert len(cross_outputs) == len(_SUBCLASS_UNRELATED_ROOTS_SEEDS)
    assert len(same_outputs) == len(_SUBCLASS_UNRELATED_ROOTS_SEEDS)

    for seed, output in zip(_SUBCLASS_UNRELATED_ROOTS_SEEDS, cross_outputs):
        assert output["outcome"] == "rejected", (
            f"cross-framework PYTHONHASHSEED={seed} should be rejected with a missing-Links error: {output}"
        )
        assert "depends on parents from" in output["error"], (
            f"cross-framework PYTHONHASHSEED={seed} should hit the plan-time rejection, not a runtime "
            f"KeyError: {output}"
        )
        assert "unlinked sources (missing Links)" in output["error"], (
            f"cross-framework PYTHONHASHSEED={seed} should hit the plan-time rejection, not a runtime "
            f"KeyError: {output}"
        )
        assert "ScRootA" in output["error"], f"cross-framework PYTHONHASHSEED={seed}: {output}"
        assert "ScRootB" in output["error"], f"cross-framework PYTHONHASHSEED={seed}: {output}"

    for seed, output in zip(_SUBCLASS_UNRELATED_ROOTS_SEEDS, same_outputs):
        assert output["outcome"] == "rejected", (
            f"same-framework PYTHONHASHSEED={seed} should be rejected with a missing-Links error: {output}"
        )
        assert "depends on parents from" in output["error"], (
            f"same-framework PYTHONHASHSEED={seed} should hit the plan-time rejection, not a runtime KeyError: {output}"
        )
        assert "unlinked sources (missing Links)" in output["error"], (
            f"same-framework PYTHONHASHSEED={seed} should hit the plan-time rejection, not a runtime KeyError: {output}"
        )
        assert "ScRootA" in output["error"], f"same-framework PYTHONHASHSEED={seed}: {output}"
        assert "ScRootB" in output["error"], f"same-framework PYTHONHASHSEED={seed}: {output}"


# Unlinked parents must be rejected whatever frameworks run, not only when a transform hop is involved.


class _Root(FeatureGroup):
    """Data-creator root: DATA maps column name to values, built in FRAMEWORK's native type."""

    DATA: ClassVar[dict[str, list[int]]] = {}
    FRAMEWORK: ClassVar[type[ComputeFramework]] = PyArrowTable

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(set(cls.DATA))

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if cls.FRAMEWORK is PandasDataFrame:
            import pandas as pd

            return pd.DataFrame(cls.DATA)
        import pyarrow as pa

        return pa.table(cls.DATA)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {cls.FRAMEWORK}


class _Consumer(FeatureGroup):
    """Consumer: OUTPUT is the sum of the INPUTS columns plus ADD, computed in FRAMEWORK's native type."""

    INPUTS: ClassVar[tuple[str, ...]] = ()
    OUTPUT: ClassVar[str] = ""
    ADD: ClassVar[int] = 0
    FRAMEWORK: ClassVar[type[ComputeFramework]] = PyArrowTable

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name) for name in self.INPUTS}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if cls.FRAMEWORK is PandasDataFrame:
            total = sum(data[name] for name in cls.INPUTS) + cls.ADD
            data[cls.OUTPUT] = total
            return data
        import pyarrow.compute as pc

        total = data[cls.INPUTS[0]]
        for name in cls.INPUTS[1:]:
            total = pc.add(total, data[name])
        return data.append_column(cls.OUTPUT, pc.add(total, cls.ADD))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {cls.FRAMEWORK}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.OUTPUT} if cls.OUTPUT else set()


class UnlinkedPaRootA(_Root):
    DATA = {"unlinked_pa_root_a": [1, 2, 3]}


class UnlinkedPaRootB(_Root):
    DATA = {"unlinked_pa_root_b": [10, 20, 30]}


class UnlinkedPaRootC(_Root):
    DATA = {"unlinked_pa_root_c": [100, 200, 300]}


class UnlinkedPaConsumer(_Consumer):
    INPUTS = ("unlinked_pa_root_a", "unlinked_pa_root_b")
    OUTPUT = "unlinked_pa_consumer_result"


class UnlinkedPaTripleConsumer(_Consumer):
    INPUTS = ("unlinked_pa_root_a", "unlinked_pa_root_b", "unlinked_pa_root_c")
    OUTPUT = "unlinked_pa_triple_result"


class UnlinkedMixPandasRoot(_Root):
    DATA = {"unlinked_mix_pd_root": [10, 20, 30]}
    FRAMEWORK = PandasDataFrame


class UnlinkedMixConsumer(_Consumer):
    INPUTS = ("unlinked_pa_root_a", "unlinked_mix_pd_root")
    OUTPUT = "unlinked_mix_result"


class SubPaRootA(_Root):
    DATA = {"sub_pa_root_a": [1, 2, 3]}


class SubPaRootB(SubPaRootA):
    DATA = {"sub_pa_root_b": [10, 20, 30]}


class SubMixPdRootB(SubPaRootA):
    DATA = {"sub_mix_pd_root_b": [10, 20, 30]}
    FRAMEWORK = PandasDataFrame


class SubPaConsumer(_Consumer):
    INPUTS = ("sub_pa_root_a", "sub_pa_root_b")
    OUTPUT = "sub_pa_result"


class SubMixConsumer(_Consumer):
    INPUTS = ("sub_pa_root_a", "sub_mix_pd_root_b")
    OUTPUT = "sub_mix_result"


_UNLINKED_CASES: list[tuple[str, set[type[FeatureGroup]], list[type[ComputeFramework]], list[str]]] = [
    (
        "unlinked_pa_consumer_result",
        {UnlinkedPaRootA, UnlinkedPaRootB, UnlinkedPaConsumer},
        [PyArrowTable],
        ["UnlinkedPaRootA", "UnlinkedPaRootB"],
    ),
    (
        "unlinked_pa_triple_result",
        {UnlinkedPaRootA, UnlinkedPaRootB, UnlinkedPaRootC, UnlinkedPaTripleConsumer},
        [PyArrowTable],
        ["UnlinkedPaRootA", "UnlinkedPaRootB", "UnlinkedPaRootC"],
    ),
    (
        "unlinked_mix_result",
        {UnlinkedPaRootA, UnlinkedMixPandasRoot, UnlinkedMixConsumer},
        [PandasDataFrame, PyArrowTable],
        ["UnlinkedPaRootA", "UnlinkedMixPandasRoot"],
    ),
    (
        "sub_pa_result",
        {SubPaRootA, SubPaRootB, SubPaConsumer},
        [PyArrowTable],
        ["SubPaRootA", "SubPaRootB"],
    ),
    (
        "sub_mix_result",
        {SubPaRootA, SubMixPdRootB, SubMixConsumer},
        [PandasDataFrame, PyArrowTable],
        ["SubPaRootA", "SubMixPdRootB"],
    ),
]


@pytest.mark.parametrize(
    "feature_name,groups,frameworks,named",
    _UNLINKED_CASES,
    ids=[
        "one_framework_two_roots",
        "one_framework_three_roots",
        "mixed_hop_and_no_hop",
        "subclass_roots_one_framework",
        "subclass_roots_mixed",
    ],
)
def test_unlinked_parents_raise_missing_links_error_regardless_of_framework(
    feature_name: str,
    groups: set[type[FeatureGroup]],
    frameworks: list[type[ComputeFramework]],
    named: list[str],
) -> None:
    with pytest.raises(ValueError) as exc_info:
        mloda.prepare(
            features=[Feature(feature_name)],
            links=set(),
            compute_frameworks=frameworks,
            plugin_collector=PluginCollector.enabled_feature_groups(groups),
        )

    error_message = str(exc_info.value)
    assert "depends on parents from" in error_message
    assert "unlinked sources (missing Links)" in error_message
    assert f"{len(named)} unlinked sources" in error_message
    assert "two different" not in error_message
    assert "Link" in error_message
    for name in named:
        assert name in error_message


# Regression guards: shapes that share one root or one class must keep planning and running.
def _column_values(results: list[Any], column: str) -> list[int]:
    for res in results:
        names = list(res.column_names) if hasattr(res, "column_names") else list(res.columns)
        if column in names:
            col = res[column]
            return [int(v) for v in (col.to_pylist() if hasattr(col, "to_pylist") else col.tolist())]
    raise AssertionError(f"column {column} not found in results")


def _run(feature_name: str, groups: set[type[FeatureGroup]], frameworks: list[type[ComputeFramework]]) -> list[Any]:
    return mloda.run_all(
        [Feature(feature_name)],
        compute_frameworks=frameworks,
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
    )


class DiamondRoot(_Root):
    DATA = {"diamond_a": [1, 2, 3]}


class DiamondD1(_Consumer):
    INPUTS = ("diamond_a",)
    OUTPUT = "diamond_d1"
    ADD = 100


class DiamondD2(_Consumer):
    INPUTS = ("diamond_a",)
    OUTPUT = "diamond_d2"
    ADD = 200


class DiamondConsumer(_Consumer):
    INPUTS = ("diamond_d1", "diamond_d2")
    OUTPUT = "diamond_result"


def test_same_root_diamond_in_one_framework_plans_and_runs() -> None:
    results = _run("diamond_result", {DiamondRoot, DiamondD1, DiamondD2, DiamondConsumer}, [PyArrowTable])
    assert _column_values(results, "diamond_result") == [302, 304, 306]


class RootDerivedRoot(_Root):
    DATA = {"rd_x": [1, 2, 3], "rd_y": [10, 20, 30]}


class RootDerivedDX(_Consumer):
    INPUTS = ("rd_x",)
    OUTPUT = "rd_dx"
    ADD = 1000


class RootDerivedConsumer(_Consumer):
    INPUTS = ("rd_dx", "rd_y")
    OUTPUT = "rd_result"


def test_root_plus_derived_in_one_framework_plans_and_runs() -> None:
    results = _run("rd_result", {RootDerivedRoot, RootDerivedDX, RootDerivedConsumer}, [PyArrowTable])
    assert _column_values(results, "rd_result") == [1011, 1022, 1033]


class HopRoot(_Root):
    DATA = {"hop_x": [1, 2, 3], "hop_y": [10, 20, 30]}
    FRAMEWORK = PandasDataFrame


class HopDX(_Consumer):
    INPUTS = ("hop_x",)
    OUTPUT = "hop_dx"
    ADD = 1000


class HopConsumer(_Consumer):
    INPUTS = ("hop_dx", "hop_y")
    OUTPUT = "hop_result"
    FRAMEWORK = PandasDataFrame


def test_same_root_with_one_hop_plans_and_runs() -> None:
    results = _run("hop_result", {HopRoot, HopDX, HopConsumer}, [PandasDataFrame, PyArrowTable])
    assert _column_values(results, "hop_result") == [1011, 1022, 1033]


class FanInRoot(_Root):
    DATA = {"fan_p": [1, 2, 3], "fan_q": [10, 20, 30]}


class FanInConsumer(_Consumer):
    INPUTS = ("fan_p", "fan_q")
    OUTPUT = "fan_result"


def test_same_class_fan_in_in_one_framework_plans_and_runs() -> None:
    results = _run("fan_result", {FanInRoot, FanInConsumer}, [PyArrowTable])
    assert _column_values(results, "fan_result") == [11, 22, 33]


# A parent that reaches a join side only through a compute-framework hop is not join-bridged, so the
# consumer must hit the plan-time missing-Links error under every hash seed, never a runtime crash.
_LINK_SIDE_PATHS_PROBE = Path(__file__).with_name("link_side_paths_probe.py")
_LINK_SIDE_PATHS_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.fixture(scope="module")
def link_side_paths_outputs() -> list[dict[str, str]]:
    return run_probes(_LINK_SIDE_PATHS_PROBE, len(_LINK_SIDE_PATHS_SEEDS), seeds=_LINK_SIDE_PATHS_SEEDS)


@pytest.mark.timeout(60)
def test_hop_parent_of_a_join_side_rejects_missing_links_under_every_hash_seed(
    link_side_paths_outputs: list[dict[str, str]],
) -> None:
    outputs = link_side_paths_outputs

    assert len(outputs) == len(_LINK_SIDE_PATHS_SEEDS)
    for seed, output in zip(_LINK_SIDE_PATHS_SEEDS, outputs):
        assert output["hop_parent_outcome"] == "rejected", f"PYTHONHASHSEED={seed}: {output}"
        assert "depends on parents from 2 unlinked sources" in output["hop_parent_error"], (
            f"PYTHONHASHSEED={seed} should hit the plan-time rejection: {output}"
        )


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", ["twin_sibling", "twin_chain"])
def test_consumers_of_a_join_side_plan_and_run_correctly_under_every_hash_seed(
    shape: str, link_side_paths_outputs: list[dict[str, str]]
) -> None:
    outputs = link_side_paths_outputs

    assert len(outputs) == len(_LINK_SIDE_PATHS_SEEDS)
    for seed, output in zip(_LINK_SIDE_PATHS_SEEDS, outputs):
        assert output[f"{shape}_outcome"] == "accepted", f"PYTHONHASHSEED={seed}: {output}"
        assert output.get(f"{shape}_values") == "[[21, 42, 63]]", f"PYTHONHASHSEED={seed}: {output}"


_HOP_DIAMOND_EXPECTED_VALUES = {
    "hop_diamond": "[[20, 40, 60]]",
    "hop_diamond_threading": "[[20, 40, 60]]",
    "hop_diamond_link": "[[21, 42, 63]]",
    "hop_diamond_link_threading": "[[21, 42, 63]]",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", list(_HOP_DIAMOND_EXPECTED_VALUES))
def test_same_root_diamond_with_a_hop_runs_correctly_under_every_hash_seed(
    shape: str, link_side_paths_outputs: list[dict[str, str]]
) -> None:
    assert len(link_side_paths_outputs) == len(_LINK_SIDE_PATHS_SEEDS)
    for seed, output in zip(_LINK_SIDE_PATHS_SEEDS, link_side_paths_outputs):
        assert output[f"{shape}_outcome"] == "accepted", f"PYTHONHASHSEED={seed}: {output}"
        assert output.get(f"{shape}_values") == _HOP_DIAMOND_EXPECTED_VALUES[shape], f"PYTHONHASHSEED={seed}: {output}"


@pytest.mark.timeout(60)
def test_same_root_diamond_mirror_is_rejected_with_its_real_cause_under_every_hash_seed(
    link_side_paths_outputs: list[dict[str, str]],
) -> None:
    first = link_side_paths_outputs[0].get("hop_diamond_mirror_error")
    for seed, output in zip(_LINK_SIDE_PATHS_SEEDS, link_side_paths_outputs):
        assert output["hop_diamond_mirror_outcome"] == "rejected", f"PYTHONHASHSEED={seed}: {output}"
        assert output["hop_diamond_mirror_error"] == first, f"PYTHONHASHSEED={seed}"
    assert "nothing merges" in str(first)
    assert "missing Links" not in str(first)


@pytest.mark.timeout(60)
def test_same_root_diamond_with_a_cycle_has_one_outcome_under_every_hash_seed(
    link_side_paths_outputs: list[dict[str, str]],
) -> None:
    keys = ("outcome", "error", "values")
    first = {key: link_side_paths_outputs[0].get(f"hop_diamond_cycle_{key}") for key in keys}
    for seed, output in zip(_LINK_SIDE_PATHS_SEEDS, link_side_paths_outputs):
        found = {key: output.get(f"hop_diamond_cycle_{key}") for key in keys}
        assert found == first, f"PYTHONHASHSEED={seed}: {found} differs from {first}"
    assert first["outcome"] in ("accepted", "rejected"), first
    assert "missing Links" not in str(first["error"])
    assert "unlinked sources" not in str(first["error"])


@pytest.mark.timeout(30)
def test_same_root_diamond_with_a_hop_runs_with_multiprocessing(flight_server: Any) -> None:
    results = mloda.run_all(
        [Feature("hd_c")],
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups(hd_probe.HD_DIAMOND_GROUPS),
        parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        flight_server=flight_server,
    )

    assert [v for v in (hd_probe._column_list(r, "hd_c") for r in results) if v is not None] == [[20, 40, 60]]
