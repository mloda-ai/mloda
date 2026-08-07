"""A RIGHT join binds the declared left group as the left merge argument; surplus left keys expose a swap."""

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


LEFT_KEYS = ["k3", "k2", "k1"]
LEFT_PAYLOADS = ["l3", "l2", "l1"]
RIGHT_KEYS = ["k1", "k2", "k9"]
RIGHT_PAYLOADS = ["r1", "r2", "r9"]

MISSING = "-"

# k3 has no right partner and drops out; k9 has no left partner and survives null-padded.
RIGHT_JOIN_ROWS = ["k1|l1|k1|r1", "k2|l2|k2|r2", f"{MISSING}|{MISSING}|k9|r9"]

SIBLING_KEYS = RIGHT_KEYS
SIBLING_PAYLOADS = ["s1", "s2", "s9"]

# The sibling covers every right key, so its inner join only widens the rows the RIGHT join already produced.
RIGHT_JOIN_ROWS_WITH_SIBLING = ["k1|l1|k1|r1|s1", "k2|l2|k2|r2|s2", f"{MISSING}|{MISSING}|k9|r9|s9"]

MODES = pytest.mark.parametrize("modes", [{ParallelizationMode.SYNC}, {ParallelizationMode.THREADING}])


def _cell(value: Any) -> str:
    """Render one joined cell so an unmatched value reads the same in every framework."""
    if value is None:
        return MISSING
    text = str(value)
    return MISSING if text in ("nan", "None", "<NA>") else text


def _columns(data: Any) -> dict[str, list[Any]]:
    if hasattr(data, "column_names"):
        return {name: data.column(name).to_pylist() for name in data.column_names}
    return {name: list(data[name]) for name in data.columns}


def _joined_rows(data: Any, prefix: str) -> list[str]:
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


def _packed_rows(results: Any, column: str) -> list[str]:
    matching = [frame for frame in results if column in _columns(frame)]
    assert len(matching) == 1, f"Expected exactly one result frame carrying {column}, got {len(matching)}."
    return [str(value) for value in _columns(matching[0])[column]]


class RightBindLeftInArrow(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"rjsb_left_key", "rjsb_left_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rjsb_left_key": LEFT_KEYS, "rjsb_left_payload": LEFT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class RightBindRightInPandas(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"rjsb_right_key", "rjsb_right_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rjsb_right_key": RIGHT_KEYS, "rjsb_right_payload": RIGHT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class RightBindChild(FeatureGroup):
    """Runs in the declared right group's framework, which is where the RIGHT join executes."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return _pair_features("rjsb")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): _joined_rows(data, "rjsb")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class RightBindPolyBase(FeatureGroup):
    """Declared left side of a link whose declared right side derives from it."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"rjsbp_left_key", "rjsbp_left_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rjsbp_left_key": LEFT_KEYS, "rjsbp_left_payload": LEFT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class RightBindPolyDerived(RightBindPolyBase):
    """Declared right side, and a subclass of the declared left side, so it answers to both sides."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"rjsbp_right_key", "rjsbp_right_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rjsbp_right_key": RIGHT_KEYS, "rjsbp_right_payload": RIGHT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class RightBindPolyChild(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return _pair_features("rjsbp")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): _joined_rows(data, "rjsbp")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class RightBindAncestorBase(FeatureGroup):
    """Declared left side of the RIGHT join, and the base class of an unrelated ancestor of the same child."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"rjsba_left_key", "rjsba_left_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rjsba_left_key": LEFT_KEYS, "rjsba_left_payload": LEFT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class RightBindAncestorRight(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"rjsba_right_key", "rjsba_right_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rjsba_right_key": RIGHT_KEYS, "rjsba_right_payload": RIGHT_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class RightBindAncestorSibling(RightBindAncestorBase):
    """Subclasses the declared left side but runs in the destination framework and joins on its own link."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"rjsba_sibling_key", "rjsba_sibling_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rjsba_sibling_key": SIBLING_KEYS, "rjsba_sibling_payload": SIBLING_PAYLOADS}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class RightBindAncestorChild(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return _pair_features("rjsba") | {Feature(name="rjsba_sibling_payload")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        siblings = _columns(data)["rjsba_sibling_payload"]
        rows = [f"{row}|{_cell(value)}" for row, value in zip(_joined_rows(data, "rjsba"), siblings)]
        return {cls.get_class_name(): rows}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


# MULTIPROCESSING is left out: cross-framework joins fail in the transform hop for unrelated reasons.
@MODES
def test_right_join_keeps_every_right_row_and_drops_unmatched_left_rows(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    link = Link.right(
        JoinSpec(RightBindLeftInArrow, Index(("rjsb_left_key",))),
        JoinSpec(RightBindRightInPandas, Index(("rjsb_right_key",))),
    )

    results = mloda.run_all(
        [Feature(name=RightBindChild.get_class_name())],
        links={link},
        compute_frameworks={PandasDataFrame, PyArrowTable},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {RightBindLeftInArrow, RightBindRightInPandas, RightBindChild}
        ),
        flight_server=flight_server if ParallelizationMode.MULTIPROCESSING in modes else None,
        parallelization_modes=modes,
    )
    rows = _packed_rows(results, RightBindChild.get_class_name())

    assert len(rows) == len(RIGHT_KEYS)
    assert sorted(rows) == sorted(RIGHT_JOIN_ROWS)


@MODES
def test_right_join_binds_the_declared_left_side_when_the_declared_right_side_subclasses_it(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    """The right node answers both issubclass tests, which must not cost it its right-argument position."""
    link = Link.right(
        JoinSpec(RightBindPolyBase, Index(("rjsbp_left_key",))),
        JoinSpec(RightBindPolyDerived, Index(("rjsbp_right_key",))),
    )

    results = mloda.run_all(
        [Feature(name=RightBindPolyChild.get_class_name())],
        links={link},
        compute_frameworks={PandasDataFrame, PyArrowTable},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {RightBindPolyBase, RightBindPolyDerived, RightBindPolyChild}
        ),
        flight_server=flight_server if ParallelizationMode.MULTIPROCESSING in modes else None,
        parallelization_modes=modes,
    )
    rows = _packed_rows(results, RightBindPolyChild.get_class_name())

    assert len(rows) == len(RIGHT_KEYS)
    assert sorted(rows) == sorted(RIGHT_JOIN_ROWS)


@MODES
def test_right_join_binds_the_declared_left_side_when_a_sibling_subclass_is_a_second_ancestor(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    """A subclass of the declared left side running in the destination framework must not rebind the sides."""
    link = Link.right(
        JoinSpec(RightBindAncestorBase, Index(("rjsba_left_key",))),
        JoinSpec(RightBindAncestorRight, Index(("rjsba_right_key",))),
    )
    sibling_link = Link.inner(
        JoinSpec(RightBindAncestorSibling, Index(("rjsba_sibling_key",))),
        JoinSpec(RightBindAncestorRight, Index(("rjsba_right_key",))),
    )

    results = mloda.run_all(
        [Feature(name=RightBindAncestorChild.get_class_name())],
        links={link, sibling_link},
        compute_frameworks={PandasDataFrame, PyArrowTable},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {RightBindAncestorBase, RightBindAncestorRight, RightBindAncestorSibling, RightBindAncestorChild}
        ),
        flight_server=flight_server if ParallelizationMode.MULTIPROCESSING in modes else None,
        parallelization_modes=modes,
    )
    rows = _packed_rows(results, RightBindAncestorChild.get_class_name())

    assert len(rows) == len(RIGHT_KEYS)
    assert sorted(rows) == sorted(RIGHT_JOIN_ROWS_WITH_SIBLING)
