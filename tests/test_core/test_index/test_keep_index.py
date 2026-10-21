"""Feature(keep_index=True) keeps the key columns next to the requested feature in its result frame.

Source groups are column-selective (return only the requested columns), so the key must really be loaded.
"""

from __future__ import annotations

import os
from typing import Any

import pyarrow as pa
import pytest

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import DataAccessCollection, Feature, FeatureName, Index, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG

KEY = "ki_id"
KEY2 = "ki_k2"
DATA: dict[str, list[int]] = {KEY: [1, 2, 3], KEY2: [10, 20, 30], "ki_val": [5, 6, 7]}

FRAMEWORKS = ["pandas", "polars", "pyarrow"]


def _fw(kind: str) -> type[ComputeFramework]:
    if kind == "polars":
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

        return PolarsDataFrame
    return PandasDataFrame if kind == "pandas" else PyArrowTable


def _frame(kind: str, data: dict[str, list[Any]]) -> Any:
    if kind == "pandas":
        import pandas as pd

        return pd.DataFrame(data)
    if kind == "polars":
        import polars as pl

        return pl.DataFrame(data)
    return pa.table(data)


def _values(kind: str, frame: Any, name: str) -> list[Any]:
    if kind == "pandas":
        return list(frame[name].tolist())
    if kind == "polars":
        return list(frame[name].to_list())
    return list(frame[name].to_pylist())


def _with(kind: str, frame: Any, name: str, values: list[Any]) -> Any:
    if kind == "pandas":
        frame = frame.copy()
        frame[name] = values
        return frame
    if kind == "polars":
        import polars as pl

        return frame.with_columns(pl.Series(name, values))
    return frame.append_column(name, pa.array(values))


def _keep_rows(kind: str, frame: Any, mask: list[bool]) -> Any:
    if kind == "pandas":
        return frame[mask].reset_index(drop=True)
    if kind == "polars":
        import polars as pl

        return frame.filter(pl.Series(mask))
    return frame.filter(pa.array(mask))


def _as_dict(kind: str, frame: Any) -> dict[str, list[Any]]:
    names = frame.columns if kind != "pyarrow" else frame.column_names
    return {str(n): _values(kind, frame, str(n)) for n in names}


def _source(kind: str, indexes: list[Index] | None) -> type[FeatureGroup]:
    class KiSource(FeatureGroup):
        @classmethod
        def input_data(cls) -> BaseInputData | None:
            return DataCreator(supports_features=set(DATA))

        @classmethod
        def index_columns(cls) -> list[Index] | None:
            return indexes

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {_fw(kind)}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return _frame(kind, {n: DATA[n] for n in features.get_all_names()})

    return KiSource


def _derived(kind: str, name: str, inputs: set[Feature], how: str = "keep") -> type[FeatureGroup]:
    class KiDerived(FeatureGroup):
        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            return str(feature_name) == name

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return inputs

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {_fw(kind)}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            vals = [v * 2 for v in _values(kind, data, "ki_val")]
            if how == "narrow":
                return _frame(kind, {name: vals})
            if how == "drop":
                data = _keep_rows(kind, data, [v > 5 for v in _values(kind, data, "ki_val")])
                return _with(kind, data, name, [v * 2 for v in _values(kind, data, "ki_val")])
            if how == "combo":
                vals = [a + b for a, b in zip(_values(kind, data, "ki_val"), _values(kind, data, KEY2))]
            return _with(kind, data, name, vals)

    return KiDerived


def _run(kind: str, features: list[Any], groups: set[type[FeatureGroup]], **kwargs: Any) -> list[Any]:
    return list(
        mloda.run_all(
            features,
            compute_frameworks=[_fw(kind)],
            plugin_collector=PluginCollector.enabled_feature_groups(groups),
            **kwargs,
        )
    )


def _dicts(kind: str, results: list[Any]) -> list[dict[str, list[Any]]]:
    return [_as_dict(kind, r) for r in results]


@pytest.mark.parametrize("kind", FRAMEWORKS)
class TestKeepIndex:
    def test_root_feature_keeps_group_index(self, kind: str) -> None:
        src = _source(kind, [Index((KEY,))])
        res = _run(kind, [Feature("ki_val", keep_index=True)], {src})
        assert _dicts(kind, res) == [{KEY: [1, 2, 3], "ki_val": [5, 6, 7]}]

    def test_derived_feature_with_feature_index(self, kind: str) -> None:
        src = _source(kind, None)
        der = _derived(kind, "ki_doubled", {Feature("ki_val")})
        f = Feature("ki_doubled", index=Index((KEY,)), keep_index=True)
        res = _dicts(kind, _run(kind, [f], {src, der}))
        assert len(res) == 1
        assert res[0][KEY] == [1, 2, 3]
        assert res[0]["ki_doubled"] == [10, 12, 14]

    def test_row_dropping_step_keeps_matching_keys(self, kind: str) -> None:
        src = _source(kind, [Index((KEY,))])
        der = _derived(kind, "ki_dropped", {Feature("ki_val")}, how="drop")
        res = _dicts(kind, _run(kind, [Feature("ki_dropped", index=Index((KEY,)), keep_index=True)], {src, der}))
        assert len(res) == 1
        assert res[0][KEY] == [2, 3]
        assert res[0]["ki_dropped"] == [12, 14]

    def test_default_output_is_unchanged(self, kind: str) -> None:
        src = _source(kind, [Index((KEY,))])
        der = _derived(kind, "ki_doubled", {Feature("ki_val")})
        res = _dicts(kind, _run(kind, [Feature("ki_doubled")], {src, der}))
        assert res == [{"ki_doubled": [10, 12, 14]}]

    def test_missing_key_raises_naming_feature_and_column(self, kind: str) -> None:
        src = _source(kind, [Index((KEY,))])
        der = _derived(kind, "ki_narrow", {Feature("ki_val")}, how="narrow")
        with pytest.raises(ValueError, match=rf"ki_narrow.*{KEY}|{KEY}.*ki_narrow"):
            _run(kind, [Feature("ki_narrow", index=Index((KEY,)), keep_index=True)], {src, der})

    @pytest.mark.parametrize("derived", [False, True], ids=["root", "derived"])
    def test_keep_index_without_any_key_raises_at_plan_time(self, kind: str, derived: bool) -> None:
        src = _source(kind, [Index((KEY,))] if derived else None)
        der = _derived(kind, "ki_doubled", {Feature("ki_val")})
        name = "ki_doubled" if derived else "ki_val"
        with pytest.raises(ValueError, match=name):
            _run(kind, [Feature(name, keep_index=True)], {src, der})

    def test_multi_index_group_keeps_all_key_columns(self, kind: str) -> None:
        src = _source(kind, [Index((KEY,)), Index((KEY2,))])
        res = _dicts(kind, _run(kind, [Feature("ki_val", keep_index=True)], {src}))
        assert res == [{KEY: [1, 2, 3], KEY2: [10, 20, 30], "ki_val": [5, 6, 7]}]

    @pytest.mark.parametrize("keep_first", [False, True], ids=["chain_first", "keep_first"])
    def test_earlier_request_subtree_already_uses_the_feature(self, kind: str, keep_first: bool) -> None:
        src = _source(kind, [Index((KEY,))])
        der = _derived(kind, "ki_doubled", {Feature("ki_val")})
        top = _derived(kind, "ki_top", {Feature("ki_doubled")}, how="top")
        keep = Feature("ki_doubled", index=Index((KEY,)), keep_index=True)
        reqs: list[Any] = [keep, "ki_top"] if keep_first else ["ki_top", keep]
        res = _dicts(kind, _run(kind, reqs, {src, der, top}))
        assert any(d.get(KEY) == [1, 2, 3] and "ki_doubled" in d for d in res)

    def test_reused_feature_instance_leaves_no_stale_key(self, kind: str) -> None:
        src = _source(kind, [Index((KEY,))])
        der = _derived(kind, "ki_doubled", {Feature("ki_val")})
        f = Feature("ki_doubled", index=Index((KEY,)), keep_index=True)
        first = _dicts(kind, _run(kind, [f], {src, der}, copy_features=False))
        assert KEY in first[0]
        f.keep_index = False
        second = _dicts(kind, _run(kind, [f], {src, der}, copy_features=False))
        assert second == [{"ki_doubled": [10, 12, 14]}]


def test_column_selective_csv_loads_the_key() -> None:
    file_path = f"{os.getcwd()}/tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"

    class KiCsv(CsvFG):
        @classmethod
        def index_columns(cls) -> list[Index] | None:
            return [Index(("id",))]

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if options.get("ki_csv") is None:
                return False
            return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    result = mloda.run_all(
        [Feature("V1", options={"ki_csv": True}, keep_index=True)],
        compute_frameworks=["PyArrowTable"],
        plugin_collector=PluginCollector.enabled_feature_groups({KiCsv}),
        data_access_collection=DataAccessCollection(files={file_path}),
    )
    assert len(result) == 1
    assert set(result[0].column_names) == {"id", "V1"}
