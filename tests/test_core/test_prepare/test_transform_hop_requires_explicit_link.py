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
