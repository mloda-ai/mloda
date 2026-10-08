"""Shared base for dict-returning sibling consumers driven through ``mloda.run_all``.

Not collected as tests (no ``Test`` prefix).
"""

from typing import Any

import pytest

from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.core.abstract_plugins.components.feature_name import FeatureName

from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import _EmptyResultMatchData
from tests.test_plugins.compute_framework.test_tooling.policy_run_all_test_base import (
    PolicyRunAllTestBase,
    PolicySuccess,
    records_from_frame,
)


def _column(data: Any, name: str) -> list[Any]:
    return [row[name] for row in records_from_frame(data)]


class SiblingDictRoot(FeatureGroup, _EmptyResultMatchData):
    """Root FeatureGroup producing the shared three-row ``sd_a`` column."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"sd_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sd_a": [1, 2, 3]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"sd_a"}


class _SiblingDictConsumer(FeatureGroup):
    """Non-root consumer of ``sd_a`` returning a dict with only its own output column."""

    FACTOR = 1

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("sd_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [v * cls.FACTOR for v in _column(data, "sd_a")]}


class SiblingDictDouble(_SiblingDictConsumer):
    FACTOR = 2


class SiblingDictTriple(_SiblingDictConsumer):
    FACTOR = 3


class SiblingDictOverlap(_SiblingDictConsumer):
    """Dict repeating the existing column plus its output (overlapping keys)."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        values = _column(data, "sd_a")
        return {"sd_a": values, cls.get_class_name(): [v + 10 for v in values]}


class SiblingDictAggregate(_SiblingDictConsumer):
    """Dict with one aggregated row (different row count than the frame)."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [sum(_column(data, "sd_a"))]}


_ALL_FGS = {
    SiblingDictRoot,
    SiblingDictDouble,
    SiblingDictTriple,
    SiblingDictOverlap,
    SiblingDictAggregate,
}
_COLLECTOR = PluginCollector.enabled_feature_groups(_ALL_FGS)


def _columns_of(result: list[Any]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    for frame in result:
        rows = records_from_frame(frame)
        for row in rows:
            for key, value in row.items():
                columns.setdefault(key, []).append(value)
    return columns


def _assert_columns(expected: dict[str, list[Any]]) -> PolicySuccess:
    def check(result: list[Any]) -> None:
        columns = _columns_of(result)
        for name, values in expected.items():
            assert name in columns, f"missing {name}, got {sorted(columns)}"
            assert sorted(columns[name]) == sorted(values)

    return PolicySuccess(assert_result=check)


class SiblingDictRunAllTestBase(PolicyRunAllTestBase):
    """Drives dict-returning sibling consumers end-to-end through ``run_all``."""

    @classmethod
    def connection_keyed_feature_groups(cls) -> set[type[FeatureGroup]]:
        return {SiblingDictRoot}

    def _run(self, names: list[str], expected: dict[str, list[Any]], mode: ParallelizationMode, server: Any) -> None:
        self.assert_policy_case(
            feature_name=names,
            plugin_collector=_COLLECTOR,
            expectation=_assert_columns(expected),
            mode=mode,
            flight_server=server,
        )

    def _run_siblings_together(self, mode: ParallelizationMode, flight_server: Any) -> None:
        self._run(
            ["SiblingDictDouble", "SiblingDictTriple"],
            {"SiblingDictDouble": [2, 4, 6], "SiblingDictTriple": [3, 6, 9]},
            mode,
            flight_server,
        )

    def _run_overlap(self, mode: ParallelizationMode, flight_server: Any) -> None:
        self._run(["SiblingDictOverlap"], {"SiblingDictOverlap": [11, 12, 13]}, mode, flight_server)

    def _run_aggregate(self, mode: ParallelizationMode, flight_server: Any) -> None:
        self._run(["SiblingDictAggregate"], {"SiblingDictAggregate": [6]}, mode, flight_server)

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    def test_sibling_dict_consumers_return_both_outputs(self, mode: ParallelizationMode, flight_server: Any) -> None:
        self._run_siblings_together(mode, flight_server)

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    def test_overlapping_dict_still_replaces_frame(self, mode: ParallelizationMode, flight_server: Any) -> None:
        self._run_overlap(mode, flight_server)

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    def test_different_row_count_dict_still_replaces_frame(self, mode: ParallelizationMode, flight_server: Any) -> None:
        self._run_aggregate(mode, flight_server)
