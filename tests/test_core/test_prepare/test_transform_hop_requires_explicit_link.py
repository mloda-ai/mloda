"""A FeatureGroupStep whose parents live on two different, unlinked source-framework instances
must raise a "missing Links" ValueError at plan-build time, not silently bind only one hop's data.
"""

from pathlib import Path
from typing import Any

import pytest

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.helpers.probe_runner import run_probes


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
            compute_frameworks={PandasDataFrame, PyArrowTable},
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
        assert "depends on parents from two different, unlinked source feature" in output["error"], (
            f"cross-framework PYTHONHASHSEED={seed} should hit the plan-time rejection, not a runtime "
            f"KeyError: {output}"
        )
        assert "ScRootA" in output["error"], f"cross-framework PYTHONHASHSEED={seed}: {output}"
        assert "ScRootB" in output["error"], f"cross-framework PYTHONHASHSEED={seed}: {output}"

    for seed, output in zip(_SUBCLASS_UNRELATED_ROOTS_SEEDS, same_outputs):
        assert output["outcome"] == "rejected", (
            f"same-framework PYTHONHASHSEED={seed} should be rejected with a missing-Links error: {output}"
        )
        assert "depends on parents from two different, unlinked source feature" in output["error"], (
            f"same-framework PYTHONHASHSEED={seed} should hit the plan-time rejection, not a runtime KeyError: {output}"
        )
        assert "ScRootA" in output["error"], f"same-framework PYTHONHASHSEED={seed}: {output}"
        assert "ScRootB" in output["error"], f"same-framework PYTHONHASHSEED={seed}: {output}"


# Unlinked parents must be rejected whatever frameworks run, not only when a transform hop is involved.


class UnlinkedPaRootA(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"unlinked_pa_root_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow as pa

        return pa.table({"unlinked_pa_root_a": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class UnlinkedPaRootB(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"unlinked_pa_root_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow as pa

        return pa.table({"unlinked_pa_root_b": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class UnlinkedPaRootC(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"unlinked_pa_root_c"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow as pa

        return pa.table({"unlinked_pa_root_c": [100, 200, 300]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class UnlinkedPaConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("unlinked_pa_root_a"), Feature("unlinked_pa_root_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("unlinked_pa_consumer_result", data.column(0))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"unlinked_pa_consumer_result"}


class UnlinkedPaTripleConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("unlinked_pa_root_a"), Feature("unlinked_pa_root_b"), Feature("unlinked_pa_root_c")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("unlinked_pa_triple_result", data.column(0))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"unlinked_pa_triple_result"}


class UnlinkedMixPandasRoot(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"unlinked_mix_pd_root"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"unlinked_mix_pd_root": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class UnlinkedMixConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("unlinked_pa_root_a"), Feature("unlinked_mix_pd_root")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("unlinked_mix_result", data.column(0))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"unlinked_mix_result"}


_UNLINKED_CASES: list[tuple[str, set[type[FeatureGroup]], set[type[ComputeFramework]], list[str]]] = [
    (
        "unlinked_pa_consumer_result",
        {UnlinkedPaRootA, UnlinkedPaRootB, UnlinkedPaConsumer},
        {PyArrowTable},
        ["UnlinkedPaRootA", "UnlinkedPaRootB"],
    ),
    (
        "unlinked_pa_triple_result",
        {UnlinkedPaRootA, UnlinkedPaRootB, UnlinkedPaRootC, UnlinkedPaTripleConsumer},
        {PyArrowTable},
        ["UnlinkedPaRootA", "UnlinkedPaRootB", "UnlinkedPaRootC"],
    ),
    (
        "unlinked_mix_result",
        {UnlinkedPaRootA, UnlinkedMixPandasRoot, UnlinkedMixConsumer},
        {PandasDataFrame, PyArrowTable},
        ["UnlinkedPaRootA", "UnlinkedMixPandasRoot"],
    ),
]


@pytest.mark.parametrize(
    "feature_name,groups,frameworks,named",
    _UNLINKED_CASES,
    ids=["one_framework_two_roots", "one_framework_three_roots", "mixed_hop_and_no_hop"],
)
def test_unlinked_parents_raise_missing_links_error_regardless_of_framework(
    feature_name: str,
    groups: set[type[FeatureGroup]],
    frameworks: set[type[ComputeFramework]],
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
    assert "depends on parents from two different, unlinked source feature" in error_message
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


def _run(feature_name: str, groups: set[type[FeatureGroup]], frameworks: set[type[ComputeFramework]]) -> list[Any]:
    return mloda.run_all(
        [Feature(feature_name)],
        compute_frameworks=frameworks,
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
    )


class DiamondRoot(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"diamond_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow as pa

        return pa.table({"diamond_a": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class DiamondD1(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("diamond_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow.compute as pc

        return data.append_column("diamond_d1", pc.add(data["diamond_a"], 100))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"diamond_d1"}


class DiamondD2(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("diamond_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow.compute as pc

        return data.append_column("diamond_d2", pc.add(data["diamond_a"], 200))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"diamond_d2"}


class DiamondConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("diamond_d1"), Feature("diamond_d2")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow.compute as pc

        return data.append_column("diamond_result", pc.add(data["diamond_d1"], data["diamond_d2"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"diamond_result"}


def test_same_root_diamond_in_one_framework_plans_and_runs() -> None:
    results = _run("diamond_result", {DiamondRoot, DiamondD1, DiamondD2, DiamondConsumer}, {PyArrowTable})
    assert _column_values(results, "diamond_result") == [302, 304, 306]


class RootDerivedRoot(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"rd_x", "rd_y"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow as pa

        return pa.table({"rd_x": [1, 2, 3], "rd_y": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class RootDerivedDX(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("rd_x")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow.compute as pc

        return data.append_column("rd_dx", pc.add(data["rd_x"], 1000))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"rd_dx"}


class RootDerivedConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("rd_dx"), Feature("rd_y")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow.compute as pc

        return data.append_column("rd_result", pc.add(data["rd_dx"], data["rd_y"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"rd_result"}


def test_root_plus_derived_in_one_framework_plans_and_runs() -> None:
    results = _run("rd_result", {RootDerivedRoot, RootDerivedDX, RootDerivedConsumer}, {PyArrowTable})
    assert _column_values(results, "rd_result") == [1011, 1022, 1033]


class HopRoot(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"hop_x", "hop_y"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"hop_x": [1, 2, 3], "hop_y": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class HopDX(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("hop_x")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow.compute as pc

        return data.append_column("hop_dx", pc.add(data["hop_x"], 1000))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"hop_dx"}


class HopConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("hop_dx"), Feature("hop_y")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["hop_result"] = data["hop_dx"] + data["hop_y"]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"hop_result"}


def test_same_root_with_one_hop_plans_and_runs() -> None:
    results = _run("hop_result", {HopRoot, HopDX, HopConsumer}, {PandasDataFrame, PyArrowTable})
    assert _column_values(results, "hop_result") == [1011, 1022, 1033]


class FanInRoot(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"fan_p", "fan_q"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow as pa

        return pa.table({"fan_p": [1, 2, 3], "fan_q": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class FanInConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("fan_p"), Feature("fan_q")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow.compute as pc

        return data.append_column("fan_result", pc.add(data["fan_p"], data["fan_q"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"fan_result"}


def test_same_class_fan_in_in_one_framework_plans_and_runs() -> None:
    results = _run("fan_result", {FanInRoot, FanInConsumer}, {PyArrowTable})
    assert _column_values(results, "fan_result") == [11, 22, 33]
