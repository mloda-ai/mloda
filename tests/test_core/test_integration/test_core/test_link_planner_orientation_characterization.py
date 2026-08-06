"""Characterizes which feature group's data ends up as the left merge argument of a link."""

from typing import Any, Optional

import pytest

from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec, Link
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

import mloda_plugins.compute_framework.base_implementations.pandas.pandas_pyarrow_transformer  # noqa: F401
import mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_pyarrow_transformer  # noqa: F401


LEFT_KEYS = ["k4", "k3", "k2", "k1"]
LEFT_PAYLOADS = ["L4", "L3", "L2", "L1"]
RIGHT_KEYS = ["k3", "k4", "k5"]
RIGHT_PAYLOADS = ["R3", "R4", "R5"]
THIRD_KEYS = ["k1", "k3", "k4", "k7"]
THIRD_PAYLOADS = ["T1", "T3", "T4", "T7"]

MISSING = "-"

# MULTIPROCESSING is opt-in per shape: the rest fail in the transform hop, on a flight the server has no table for.
MODES_SYNC_THREADING = pytest.mark.parametrize(
    "modes",
    [{ParallelizationMode.SYNC}, {ParallelizationMode.THREADING}],
)

MODES_WITH_MULTIPROCESSING = pytest.mark.parametrize(
    "modes",
    [
        {ParallelizationMode.SYNC},
        {ParallelizationMode.THREADING},
        # Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
        pytest.param({ParallelizationMode.MULTIPROCESSING}, marks=pytest.mark.timeout(30)),
    ],
)


def _cell(value: Any) -> str:
    """Render one joined cell so an unmatched value reads the same in every framework."""
    if value is None:
        return MISSING
    text = str(value)
    return MISSING if text in ("nan", "None", "<NA>") else text


def _columns(data: Any) -> dict[str, list[Any]]:
    if isinstance(data, dict):
        return {name: list(values) for name, values in data.items()}
    if hasattr(data, "column_names"):
        return {name: data.column(name).to_pylist() for name in data.column_names}
    return {name: list(data[name]) for name in data.columns}


def _pair_rows(data: Any, prefix: str) -> list[str]:
    columns = _columns(data)
    return [
        "|".join(_cell(value) for value in row)
        for row in zip(
            columns[f"{prefix}_left_key"],
            columns[f"{prefix}_left_payload"],
            columns[f"{prefix}_right_key"],
            columns[f"{prefix}_right_payload"],
        )
    ]


def _pair_features(prefix: str) -> set[Feature]:
    return {
        Feature(name=f"{prefix}_left_key"),
        Feature(name=f"{prefix}_left_payload"),
        Feature(name=f"{prefix}_right_key"),
        Feature(name=f"{prefix}_right_payload"),
    }


def _packed_frame(results: Any, column: str) -> Any:
    matching = [frame for frame in results if column in _columns(frame)]
    assert len(matching) == 1, f"Expected exactly one result frame carrying {column}, got {len(matching)}."
    return matching[0]


def _packed_rows(results: Any, column: str) -> list[str]:
    return [str(value) for value in _columns(_packed_frame(results, column))[column]]


class OrientCharLeftInPandas(FeatureGroup):
    """Declared left side of pair A, so a Pandas child joins in the declared orientation."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"oc_a_left_key", "oc_a_left_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"oc_a_left_key": LEFT_KEYS, "oc_a_left_payload": LEFT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class OrientCharRightInArrow(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"oc_a_right_key", "oc_a_right_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"oc_a_right_key": RIGHT_KEYS, "oc_a_right_payload": RIGHT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class OrientCharLeftInArrow(FeatureGroup):
    """Declared left side of pair B, so a Pandas child inverts the declared orientation."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"oc_b_left_key", "oc_b_left_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"oc_b_left_key": LEFT_KEYS, "oc_b_left_payload": LEFT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class OrientCharRightInPandas(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"oc_b_right_key", "oc_b_right_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"oc_b_right_key": RIGHT_KEYS, "oc_b_right_payload": RIGHT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class OrientCharThirdInDict(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"oc_c_key", "oc_c_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"oc_c_key": THIRD_KEYS, "oc_c_payload": THIRD_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}


class OrientCharDeclaredChild(FeatureGroup):
    """Runs in the framework of pair A's declared left parent."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return _pair_features("oc_a")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): _pair_rows(data, "oc_a")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class OrientCharInvertedChild(FeatureGroup):
    """Runs in the framework of pair B's declared right parent."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return _pair_features("oc_b")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): _pair_rows(data, "oc_b")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class OrientCharArrowChild(FeatureGroup):
    """Declares the framework of pair B's declared left parent."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return _pair_features("oc_b")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): _pair_rows(data, "oc_b")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class OrientCharChainChild(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature(name="oc_a_left_key"),
            Feature(name="oc_a_left_payload"),
            Feature(name="oc_a_right_payload"),
            Feature(name="oc_c_payload"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        columns = _columns(data)
        rows = [
            "|".join(_cell(value) for value in row)
            for row in zip(
                columns["oc_a_left_key"],
                columns["oc_a_left_payload"],
                columns["oc_a_right_payload"],
                columns["oc_c_payload"],
            )
        ]
        return {cls.get_class_name(): rows}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


PAIR_A = (OrientCharLeftInPandas, OrientCharRightInArrow, "oc_a")
PAIR_B = (OrientCharLeftInArrow, OrientCharRightInPandas, "oc_b")


def _pair_link(pair: tuple[type[FeatureGroup], type[FeatureGroup], str], jointype: str) -> Link:
    left_group, right_group, prefix = pair
    return Link(
        jointype,
        JoinSpec(left_group, Index((f"{prefix}_left_key",))),
        JoinSpec(right_group, Index((f"{prefix}_right_key",))),
    )


def _run_pair_results(
    pair: tuple[type[FeatureGroup], type[FeatureGroup], str],
    jointype: str,
    child: type[FeatureGroup],
    modes: set[ParallelizationMode],
    flight_server: Any,
) -> Any:
    left_group, right_group, _ = pair
    return mloda.run_all(
        [Feature(name=child.get_class_name())],
        links={_pair_link(pair, jointype)},
        compute_frameworks={PandasDataFrame, PyArrowTable},
        plugin_collector=PluginCollector.enabled_feature_groups({left_group, right_group, child}),
        flight_server=flight_server if ParallelizationMode.MULTIPROCESSING in modes else None,
        parallelization_modes=modes,
    )


def _run_pair(
    pair: tuple[type[FeatureGroup], type[FeatureGroup], str],
    jointype: str,
    child: type[FeatureGroup],
    modes: set[ParallelizationMode],
    flight_server: Any,
) -> list[str]:
    results = _run_pair_results(pair, jointype, child, modes, flight_server)
    return _packed_rows(results, child.get_class_name())


INNER_ROWS = ["k4|L4|k4|R4", "k3|L3|k3|R3"]
LEFT_JOIN_ROWS = INNER_ROWS + [f"k2|L2|{MISSING}|{MISSING}", f"k1|L1|{MISSING}|{MISSING}"]
RIGHT_JOIN_ROWS = ["k3|L3|k3|R3", "k4|L4|k4|R4", f"{MISSING}|{MISSING}|k5|R5"]
CHAIN_ROWS = ["k4|L4|R4|T4", "k3|L3|R3|T3"]

RIGHT_SIDE_BINDING_REASON = (
    "plain RIGHT joins bind the merge arguments to the resolved frameworks instead of the "
    "declared sides, so the declared left index is looked up in the right group's data"
)


@MODES_WITH_MULTIPROCESSING
def test_inner_join_declared_orientation_keeps_left_group_first(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    rows = _run_pair(PAIR_A, "inner", OrientCharDeclaredChild, modes, flight_server)

    assert rows == INNER_ROWS


@MODES_SYNC_THREADING
def test_inner_join_inverted_orientation_keeps_left_group_first(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    rows = _run_pair(PAIR_B, "inner", OrientCharInvertedChild, modes, flight_server)

    assert rows == INNER_ROWS


@MODES_WITH_MULTIPROCESSING
def test_left_join_declared_orientation_keeps_every_left_row(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    rows = _run_pair(PAIR_A, "left", OrientCharDeclaredChild, modes, flight_server)

    assert rows == LEFT_JOIN_ROWS


@MODES_SYNC_THREADING
def test_left_join_inverted_orientation_keeps_every_left_row(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    rows = _run_pair(PAIR_B, "left", OrientCharInvertedChild, modes, flight_server)

    assert rows == LEFT_JOIN_ROWS


@MODES_SYNC_THREADING
def test_right_join_keeps_every_right_row_for_a_child_declaring_the_left_framework(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    results = _run_pair_results(PAIR_B, "right", OrientCharArrowChild, modes, flight_server)
    frame = _packed_frame(results, OrientCharArrowChild.get_class_name())

    # The planner rewrote the child off its declared PyArrowTable, without asking whether it supports the new one.
    assert type(frame) is PandasDataFrame.expected_data_framework()
    assert _packed_rows(results, OrientCharArrowChild.get_class_name()) == RIGHT_JOIN_ROWS


@MODES_SYNC_THREADING
def test_right_join_raises_for_a_child_on_the_right_framework(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    # The exception type is incidental: the column-semantics guard reaches the key column before the merge does.
    with pytest.raises((KeyError, ValueError), match="oc_b_left_key"):
        _run_pair(PAIR_B, "right", OrientCharInvertedChild, modes, flight_server)


@pytest.mark.xfail(strict=True, reason=RIGHT_SIDE_BINDING_REASON)
@MODES_SYNC_THREADING
def test_right_join_should_keep_every_right_row_for_a_child_on_the_right_framework(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    rows = _run_pair(PAIR_B, "right", OrientCharInvertedChild, modes, flight_server)

    assert sorted(rows) == sorted(RIGHT_JOIN_ROWS)


@MODES_SYNC_THREADING
def test_chained_join_across_three_frameworks_keeps_left_group_order(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    first = JoinSpec(OrientCharLeftInPandas, Index(("oc_a_left_key",)))
    second = JoinSpec(OrientCharRightInArrow, Index(("oc_a_right_key",)))
    third = JoinSpec(OrientCharThirdInDict, Index(("oc_c_key",)))

    result = mloda.run_all(
        [Feature(name=OrientCharChainChild.get_class_name())],
        links={Link.inner(first, second), Link.inner(second, third)},
        compute_frameworks={PandasDataFrame, PyArrowTable, PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {OrientCharLeftInPandas, OrientCharRightInArrow, OrientCharThirdInDict, OrientCharChainChild}
        ),
        flight_server=flight_server if ParallelizationMode.MULTIPROCESSING in modes else None,
        parallelization_modes=modes,
    )
    rows = _packed_rows(result, OrientCharChainChild.get_class_name())

    assert rows == CHAIN_ROWS
