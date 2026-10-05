"""Prints one json line: per planner shape, the outcome a fresh interpreter plans and runs.
No test_ prefix, so pytest never collects it; add a shape by extending SHAPES.
"""

import json
from collections.abc import Callable
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, JoinSpec, Link, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_pyarrow_transformer import *  # noqa: F403
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


def _sum_columns(data: Any, name: str, columns: list[str]) -> Any:
    if hasattr(data, "iloc"):
        data[name] = sum(data[column] for column in columns)
        return data
    total = data[columns[0]]
    for column in columns[1:]:
        total = pc.add(total, data[column])
    return data.append_column(name, total)


class LsRootX(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"ls_x"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"ls_xid": [1, 2, 3], "ls_x": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("ls_xid",))]


class LsRootW(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"ls_w"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"ls_xid": [1, 2, 3], "ls_w": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("ls_xid",))]


class _SumGroup(FeatureGroup):
    """Reads INPUTS and sums them into OUTPUT; FRAMEWORK None keeps the default framework rule."""

    INPUTS: ClassVar[tuple[str, ...]] = ()
    OUTPUT: ClassVar[str] = ""
    FRAMEWORK: ClassVar[type[ComputeFramework] | None] = None

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name) for name in self.INPUTS}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _sum_columns(data, cls.OUTPUT, list(cls.INPUTS))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {cls.FRAMEWORK} if cls.FRAMEWORK else None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.OUTPUT} if cls.OUTPUT else set()


def _sum_group(
    name: str, inputs: tuple[str, ...], output: str, framework: type[ComputeFramework] | None
) -> type[_SumGroup]:
    return type(
        name, (_SumGroup,), {"INPUTS": inputs, "OUTPUT": output, "FRAMEWORK": framework, "__module__": __name__}
    )


LsD = _sum_group("LsD", ("ls_x",), "ls_d", PyArrowTable)
LsE = _sum_group("LsE", ("ls_x", "ls_w"), "ls_e", None)
LsYf = _sum_group("LsYf", ("ls_c1",), "ls_yf", PyArrowTable)


class LsC(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        if str(feature_name) == "ls_c1":
            return {Feature("ls_d")}
        return {Feature("ls_d"), Feature("ls_e"), Feature("ls_yf")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if "ls_c1" in features.get_all_names():
            data = _sum_columns(data, "ls_c1", ["ls_d"])
        if "ls_c2" in features.get_all_names():
            data = _sum_columns(data, "ls_c2", ["ls_d", "ls_e", "ls_yf"])
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ls_c1", "ls_c2"}


LsY = _sum_group("LsY", ("ls_x",), "ls_y", PyArrowTable)
LsZ = _sum_group("LsZ", ("ls_x", "ls_w"), "ls_z", PyArrowTable)
LsYzC = _sum_group("LsYzC", ("ls_y", "ls_z"), "ls_yzc", PandasDataFrame)
LsP = _sum_group("LsP", ("ls_x",), "ls_p", PyArrowTable)
LsQ = _sum_group("LsQ", ("ls_p", "ls_w"), "ls_q", PyArrowTable)
LsPqC = _sum_group("LsPqC", ("ls_p", "ls_q"), "ls_pqc", PandasDataFrame)


class HdRootA(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"hd_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"hd_rid": [1, 2, 3], "hd_a": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class HdRootB(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"hd_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"hd_rid": [1, 2, 3], "hd_b": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


HdP = _sum_group("HdP", ("hd_a",), "hd_p", PandasDataFrame)
HdQ = _sum_group("HdQ", ("hd_a",), "hd_q", PyArrowTable)
HdQJoin = _sum_group("HdQJoin", ("hd_a", "hd_b"), "hd_q", PyArrowTable)
HdQCycle = _sum_group("HdQCycle", ("hd_a", "hd_p"), "hd_q", PyArrowTable)
HdC = _sum_group("HdC", ("hd_q", "hd_p"), "hd_c", PyArrowTable)
HdCPandas = _sum_group("HdCPandas", ("hd_q", "hd_p"), "hd_c", PandasDataFrame)

HD_DIAMOND_GROUPS: set[type[FeatureGroup]] = {HdRootA, HdP, HdQ, HdC}
HD_DIAMOND_LINK_GROUPS: set[type[FeatureGroup]] = {HdRootA, HdRootB, HdP, HdQJoin, HdC}
HD_MIRROR_GROUPS: set[type[FeatureGroup]] = {HdRootA, HdP, HdQ, HdCPandas}
HD_CYCLE_GROUPS: set[type[FeatureGroup]] = {HdRootA, HdP, HdQCycle, HdC}
HD_LINK: Link = Link.inner(JoinSpec(HdRootA, "hd_rid"), JoinSpec(HdRootB, "hd_rid"))


def _column_list(result: Any, column: str) -> list[int] | None:
    names = list(result.column_names) if hasattr(result, "column_names") else list(result.columns)
    if column not in names:
        return None
    values = result[column]
    return sorted(int(v) for v in (values.to_pylist() if hasattr(values, "to_pylist") else values.tolist()))


def _run(
    requested: list[str],
    groups: set[type[FeatureGroup]],
    links: set[Link],
    column: str | None = None,
    mode: ParallelizationMode = ParallelizationMode.SYNC,
) -> dict[str, str]:
    """A plan-time reject shows up as the caught exception's message; column adds its sorted values."""
    try:
        results = mloda.run_all(
            [Feature(name) for name in requested],
            compute_frameworks=[PandasDataFrame, PyArrowTable],
            links=links,
            plugin_collector=PluginCollector.enabled_feature_groups(groups),
            parallelization_modes={mode},
        )
    except ValueError as error:
        return {"outcome": "rejected", "error": str(error)}
    except Exception as error:
        return {"outcome": "crashed", "error": f"{type(error).__name__}: {error}"}
    found = {"outcome": "accepted", "error": ""}
    if column is not None:
        found["values"] = json.dumps(
            [values for values in (_column_list(result, column) for result in results) if values is not None]
        )
    return found


def hop_parent_shape() -> dict[str, str]:
    """A hop parent reaches a join side only through a framework change, so its join is missing."""
    link = Link.inner(JoinSpec(LsRootX, "ls_xid"), JoinSpec(LsRootW, "ls_xid"))
    return _run(["ls_c2"], {LsRootX, LsRootW, LsD, LsE, LsYf, LsC}, {link})


def twin_sibling_shape() -> dict[str, str]:
    """Y and Z both read join side X, Z also reads W, and a pandas-only C reads Y and Z."""
    link = Link.inner(JoinSpec(LsRootX, "ls_xid"), JoinSpec(LsRootW, "ls_xid"))
    return _run(["ls_yzc"], {LsRootX, LsRootW, LsY, LsZ, LsYzC}, {link}, "ls_yzc")


def twin_chain_shape() -> dict[str, str]:
    """P reads join side X, Q reads P and W, and a pandas-only C reads P and Q."""
    link = Link.inner(JoinSpec(LsRootX, "ls_xid"), JoinSpec(LsRootW, "ls_xid"))
    return _run(["ls_pqc"], {LsRootX, LsRootW, LsP, LsQ, LsPqC}, {link}, "ls_pqc")


def hop_diamond_shape(mode: ParallelizationMode = ParallelizationMode.SYNC) -> dict[str, str]:
    """P (pandas) and Q (pyarrow) both read root A, a pyarrow C reads Q and P."""
    return _run(["hd_c"], HD_DIAMOND_GROUPS, set(), "hd_c", mode)


def hop_diamond_threading_shape() -> dict[str, str]:
    return hop_diamond_shape(ParallelizationMode.THREADING)


def hop_diamond_link_shape(mode: ParallelizationMode = ParallelizationMode.SYNC) -> dict[str, str]:
    """As hop_diamond, but Q is the join consumer of Link A-B."""
    return _run(["hd_c"], HD_DIAMOND_LINK_GROUPS, {HD_LINK}, "hd_c", mode)


def hop_diamond_link_threading_shape() -> dict[str, str]:
    return hop_diamond_link_shape(ParallelizationMode.THREADING)


def hop_diamond_mirror_shape() -> dict[str, str]:
    """As hop_diamond, but C is pandas, so P is same-framework and Q arrives through a hop."""
    return _run(["hd_c"], HD_MIRROR_GROUPS, set(), "hd_c")


def hop_diamond_cycle_shape() -> dict[str, str]:
    """Q reads A and P, and C reads Q and P."""
    return _run(["hd_c"], HD_CYCLE_GROUPS, set(), "hd_c")


SHAPES: dict[str, Callable[[], dict[str, str]]] = {
    "hop_parent": hop_parent_shape,
    "twin_sibling": twin_sibling_shape,
    "twin_chain": twin_chain_shape,
    "hop_diamond": hop_diamond_shape,
    "hop_diamond_threading": hop_diamond_threading_shape,
    "hop_diamond_link": hop_diamond_link_shape,
    "hop_diamond_link_threading": hop_diamond_link_threading_shape,
    "hop_diamond_mirror": hop_diamond_mirror_shape,
    "hop_diamond_cycle": hop_diamond_cycle_shape,
}


def collect() -> dict[str, str]:
    collected: dict[str, str] = {}
    for shape, build in SHAPES.items():
        for key, value in build().items():
            collected[f"{shape}_{key}"] = value
    return collected


if __name__ == "__main__":
    print(json.dumps(collect(), sort_keys=True))
