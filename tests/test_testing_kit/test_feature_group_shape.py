"""check_feature_group_shape: a conforming FeatureGroup passes, each broken one fails its own sub-check."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import pytest

from mloda._testing.framework_adapter import (
    IcebergAdapter,
    PandasAdapter,
    PolarsAdapter,
    PyArrowAdapter,
    PythonDictAdapter,
)
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda._testing.feature_group_shape import assert_feature_group_shape, check_feature_group_shape
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.implementation import Implementation
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup
from mloda.core.abstract_plugins.components.divergence import Cause, Divergence, DivergenceKind
from mloda.provider import DefaultOptionKeys, SubtypeDeclaration
from mloda.user.python_dict import PythonDictFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.environment.installed_packages_feature_group import (
    InstalledPackagesFeatureGroup,
)
from mloda_plugins.feature_group.experimental.environment.list_directory_feature_group import (
    ListDirectoryFeatureGroup,
)
from mloda_plugins.feature_group.experimental.source_input_feature import SourceInputFeature


def make_toy_group() -> type[FeatureGroup]:
    class ToyGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            "mode": PropertySpec("Required mode.", allowed_values=("fast", "slow"), strict_validation=True),
            "scale": PropertySpec("Optional scale.", default="1"),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"toy_a"}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature("toy_src")}

        @classmethod
        def return_data_type_rule(cls, feature: Feature) -> DataType | None:
            return DataType.INT64

        @classmethod
        def calculate_feature(cls, data: Any, features: Any) -> Any:
            return _double_into(data, "toy_a", "toy_src")

    return ToyGroup


def _is_columnar_dict(data: Any) -> bool:
    return isinstance(data, dict)


def _double_into(data: Any, out: str, src: str) -> Any:
    """Add out = src * 2 to a dict, pandas, polars or Arrow table frame."""
    if _is_columnar_dict(data):
        data[out] = [v * 2 for v in data[src]]
        return data
    if hasattr(data, "iloc"):
        data[out] = data[src] * 2
        return data
    if hasattr(data, "with_columns"):
        import polars as pl

        return data.with_columns((pl.col(src) * 2).alias(out))
    import pyarrow.compute as pc

    return data.append_column(out, pc.multiply(data[src], 2))


_PART_KINDS = (
    "parts_ok",
    "parts_count_dependent",
    "parts_never_b",
    "nondeterministic_part",
    "part_in_one_run",
    "plain_in_one_run",
    "wrong_type_part",
    "plain_only",
    "pivot_extra",
)


def _write_parts(kind: str, data: dict[str, list[Any]], counter: list[int]) -> dict[str, list[Any]]:
    rows = len(data["rg_src"])
    doubled = [v * 2 for v in data["rg_src"]]
    counter[0] += 1
    later_clean_run = bool(rows) and counter[0] > 1 and data["rg_label"][0] == "alpha-secret"
    if kind == "parts_ok":
        data["rg_out~a"], data["rg_out~b"] = doubled, doubled
    elif kind == "parts_count_dependent":
        data["rg_out~a"] = doubled
        if rows:
            data["rg_out~b"] = doubled
    elif kind == "parts_never_b":
        data["rg_out~a"] = doubled
    elif kind == "nondeterministic_part":
        data["rg_out~a"] = [counter[0]] * rows
    elif kind == "part_in_one_run":
        data["rg_out~a"] = doubled
        if not later_clean_run:
            data["rg_out~b"] = doubled
    elif kind == "plain_in_one_run":
        if not later_clean_run:
            data["rg_out"] = doubled
    elif kind == "wrong_type_part":
        data["rg_out~a"] = [str(v) for v in doubled]
    elif kind == "plain_only":
        data["rg_out"] = doubled
    elif kind == "pivot_extra":
        data["rg_out"] = doubled
        if rows:
            data[f"pivot_{data['rg_label'][0]}"] = doubled
    return data


def make_run_group(
    kind: str = "ok",
    rule: set[type[ComputeFramework]] | None = None,
    output_parts: tuple[str, ...] | None = None,
    names: frozenset[str] = frozenset({"rg_out"}),
) -> type[FeatureGroup]:
    """A group over input rg_src (INT64) and rg_label (STRING) that writes rg_out; kind injects one defect."""
    counter = [0]

    class RunGroup(FeatureGroup):
        if output_parts is not None:
            OUTPUT_PARTS = output_parts

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return rule

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return set(names)

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature("rg_src"), Feature("rg_label")}

        @classmethod
        def return_data_type_rule(cls, feature: Feature) -> DataType | None:
            if kind == "object_date":
                return DataType.DATE
            if kind == "tz_out":
                return DataType.TIMESTAMP_MICROS_TZ
            if kind in ("object_string", "iceberg_scan"):
                return DataType.STRING
            if kind == "list_null_on_zero_rows":
                return DataType.LIST
            return DataType.INT64

        @classmethod
        def calculate_feature(cls, data: Any, features: Any) -> Any:
            if kind in _PART_KINDS:
                return _write_parts(kind, data, counter)
            if kind == "drops_output":
                return data
            if kind == "writes_names":
                for written in sorted(names):
                    data[written] = [v * 2 for v in data["rg_src"]]
                return data
            if kind == "narrow_type":
                if hasattr(data, "iloc"):
                    data["rg_out"] = (data["rg_src"] * 2).astype("int32")
                    return data
                if hasattr(data, "with_columns"):
                    import polars as pl

                    return data.with_columns((pl.col("rg_src") * 2).cast(pl.Int32).alias("rg_out"))
                import pyarrow as pa
                import pyarrow.compute as pc

                return data.append_column("rg_out", pc.cast(pc.multiply(data["rg_src"], 2), pa.int32()))
            if kind == "int64_on_zero_rows":
                import pandas as pd

                values = [i * 2 for i in range(len(data))]
                data["rg_out"] = pd.Series(values, index=data.index, dtype="Int64" if values else "int64")
                return data
            if kind == "object_date":
                import pandas as pd
                from datetime import date

                data["rg_out"] = pd.Series([date(2024, 1, 1)] * len(data), index=data.index, dtype=object)
                return data
            if kind == "iceberg_scan":
                import pyarrow as pa

                table = data.scan().to_arrow()
                return table.append_column("rg_out", pa.array(["x"] * table.num_rows, type=pa.string()))
            if kind == "object_string":
                import pandas as pd

                data["rg_out"] = pd.Series(["x"] * len(data), index=data.index, dtype=object)
                return data
            if kind == "list_null_on_zero_rows":
                import polars as pl

                dtype = pl.List(pl.Int64) if len(data) else pl.List(pl.Null)
                return data.with_columns(pl.Series("rg_out", [[1]] * len(data), dtype=dtype))
            if kind == "tz_out":
                from datetime import UTC, datetime

                import pyarrow as pa

                stamps = [datetime(2024, 1, 1, tzinfo=UTC)] * data.num_rows
                return data.append_column("rg_out", pa.array(stamps, type=pa.timestamp("us", tz="UTC")))
            if kind == "overwrites_input":
                data = _double_into(data, "rg_out", "rg_src")
                data["rg_src"] = [0 for _ in data["rg_src"]]
                return data
            if kind == "wrong_type":
                if _is_columnar_dict(data):
                    data["rg_out"] = [str(v) for v in data["rg_src"]]
                    return data
                if hasattr(data, "iloc"):
                    data["rg_out"] = data["rg_src"].astype("string")
                    return data
                import pyarrow as pa

                strings = [str(v) for v in data["rg_src"].to_pylist()]
                return data.append_column("rg_out", pa.array(strings, type=pa.string()))
            if kind == "emits_na":
                import pandas as pd

                data["rg_out"] = pd.array([1 if i % 2 == 0 else pd.NA for i in range(len(data))], dtype="Int64")
                return data
            if kind == "loses_schema_on_zero_rows":
                import pyarrow as pa

                if data.num_rows == 0:
                    return data.append_column("rg_out", pa.nulls(0))
                return _double_into(data, "rg_out", "rg_src")
            if kind == "leaks_cell_in_error":
                labels = data["rg_label"]
                if labels and labels[0] != "alpha-secret":
                    raise ValueError(f"unexpected label {labels[0]}")
                return _double_into(data, "rg_out", "rg_src")
            if kind == "nondeterministic":
                counter[0] += 1
                data["rg_out"] = [counter[0]] * len(data["rg_src"])
                return data
            return _double_into(data, "rg_out", "rg_src")

    return RunGroup


def make_operation_group(
    names: frozenset[str] = frozenset({"rg_out"}),
    variants: frozenset[str] = frozenset(),
    subtypes: bool = False,
) -> type[FeatureGroup]:
    class OperationGroup(OperationFeatureGroup):
        if variants:
            VARIANTS = variants
        if subtypes:
            SUBTYPES = SubtypeDeclaration(key="op_kind")
            PROPERTY_MAPPING = {
                "op_kind": PropertySpec("Kind.", allowed_values=("sum", "median"), strict_validation=True)
            }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return set(names)

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature("rg_src"), Feature("rg_label")}

        @classmethod
        def return_data_type_rule(cls, feature: Feature) -> DataType | None:
            return DataType.INT64

    return OperationGroup


def make_static_run_group() -> type[FeatureGroup]:
    class StaticRunGroup(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"rg_out"}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature("rg_src"), Feature("rg_label")}

        @classmethod
        def return_data_type_rule(cls, feature: Feature) -> DataType | None:
            return DataType.INT64

        @staticmethod
        def calculate_feature(data: Any, features: Any) -> Any:
            return _double_into(data, "rg_out", "rg_src")

    return StaticRunGroup


def _run_data() -> TypedTestData:
    return TypedTestData(
        {
            "rg_src": (DataType.INT64, [1, 2, 3]),
            "rg_label": (DataType.STRING, ["alpha-secret", "beta-secret", "gamma-secret"]),
        }
    )


def make_always_matches() -> type[FeatureGroup]:
    class AlwaysMatches(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"am_a"}

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return True

    return AlwaysMatches


def make_rejects_declared_name() -> type[FeatureGroup]:
    class RejectsDeclaredName(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"rd_a", "rd_b"}

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) in {"rd_a", cls.get_class_name()}

    return RejectsDeclaredName


def make_self_input() -> type[FeatureGroup]:
    class SelfInput(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"self_in"}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature("self_in")}

    return SelfInput


def make_ignores_strict_values() -> type[FeatureGroup]:
    class IgnoresStrictValues(FeatureGroup):
        PROPERTY_MAPPING = {
            "mode": PropertySpec("Mode.", allowed_values=("a", "b"), strict_validation=True, default="a"),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"is_a"}

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) in {"is_a", cls.get_class_name()}

    return IgnoresStrictValues


def make_matches_any_name_with_undeclared_option() -> type[FeatureGroup]:
    class MatchesWithUndeclaredOption(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"mu_a"}

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            if options.group or options.context:
                return True
            return str(feature_name) in {"mu_a", cls.get_class_name()}

    return MatchesWithUndeclaredOption


def make_raising_inputs() -> type[FeatureGroup]:
    class RaisingInputs(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"ri_a"}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            raise AttributeError("broken input_features")

    return RaisingInputs


def make_falsy_first_allowed_value(values: tuple[Any, ...]) -> type[FeatureGroup]:
    class FalsyFirst(FeatureGroup):
        PROPERTY_MAPPING = {
            "mode": PropertySpec("Required mode.", allowed_values=values, strict_validation=True),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"ff_a"}

    return FalsyFirst


def make_ignores_strict_for_second_name() -> type[FeatureGroup]:
    class IgnoresStrictForSecondName(FeatureGroup):
        PROPERTY_MAPPING = {
            "mode": PropertySpec("Mode.", allowed_values=("a", "b"), strict_validation=True, default="a"),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"is2_a", "is2_b"}

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            if str(feature_name) == "is2_b":
                return True
            return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    return IgnoresStrictForSecondName


def test_toy_group_passes() -> None:
    result = check_feature_group_shape(make_toy_group())
    assert result.check == "feature_group_shape"
    assert result.subject == "ToyGroup"
    assert result.failures == ()


def test_toy_group_passes_with_explicit_base_options() -> None:
    result = check_feature_group_shape(make_toy_group(), base_options=Options(group={"mode": "slow"}))
    assert result.passed


@pytest.mark.parametrize(
    ("cls", "base"),
    [
        (InstalledPackagesFeatureGroup, None),
        (ListDirectoryFeatureGroup, None),
        (SourceInputFeature, Options(group={DefaultOptionKeys.in_features: frozenset(["src_feature"])})),
    ],
)
def test_builtin_groups_pass(cls: type[FeatureGroup], base: Options | None) -> None:
    assert check_feature_group_shape(cls, base_options=base).failures == ()


@pytest.mark.parametrize("values", [("", "b"), (0, 1)])
def test_falsy_first_allowed_value_is_buildable(values: tuple[Any, ...]) -> None:
    result = check_feature_group_shape(make_falsy_first_allowed_value(values))
    assert not any("cannot build" in f for f in result.failures), result.failures
    assert result.passed, result.failures


def test_undeclared_option_failure_is_named() -> None:
    result = check_feature_group_shape(make_matches_any_name_with_undeclared_option())
    assert any("undeclared option" in f for f in result.failures), result.failures


@pytest.mark.parametrize(
    ("make", "prefix"),
    [
        (make_always_matches, "foreign:"),
        (make_rejects_declared_name, "names:"),
        (make_self_input, "inputs:"),
        (make_ignores_strict_values, "strict:"),
        (make_matches_any_name_with_undeclared_option, "foreign:"),
        (make_raising_inputs, "inputs:"),
        (make_ignores_strict_for_second_name, "strict:"),
    ],
)
def test_broken_group_fails_only_its_sub_check(make: Callable[[], type[FeatureGroup]], prefix: str) -> None:
    result = check_feature_group_shape(make())
    assert not result.passed
    assert all(f.startswith(prefix) for f in result.failures), result.failures


def test_unbuildable_request_is_reported_not_raised() -> None:
    class NoDerivableKey(FeatureGroup):
        PROPERTY_MAPPING = {"free": PropertySpec("Free-form required key.")}

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"nd_a"}

    result = check_feature_group_shape(NoDerivableKey)
    assert any(f.startswith("names:") and "cannot build a matching request" in f for f in result.failures)


def test_assert_passes_silently_for_conforming_group() -> None:
    assert_feature_group_shape(make_toy_group())


def test_assert_raises_pytest_failure_for_broken_group() -> None:
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_feature_group_shape(make_always_matches())
    assert "foreign:" in str(excinfo.value)


def _require_backend(adapter_cls: type[Any]) -> None:
    if adapter_cls is not PythonDictAdapter:
        pytest.importorskip("pyarrow")
    if adapter_cls is IcebergAdapter:
        pytest.importorskip("pyiceberg")
    if adapter_cls is PandasAdapter:
        pytest.importorskip("pandas")
    if adapter_cls is PolarsAdapter:
        pytest.importorskip("polars")


def _toy_data() -> TypedTestData:
    return TypedTestData({"toy_src": (DataType.INT64, [1, 2, 3])})


@pytest.mark.parametrize("adapter_cls", [PythonDictAdapter, PyArrowAdapter, PandasAdapter, PolarsAdapter])
def test_toy_group_passes_run_checks(adapter_cls: type[Any]) -> None:
    _require_backend(adapter_cls)
    result = check_feature_group_shape(make_toy_group(), input_data=_toy_data(), adapter=adapter_cls())
    assert result.failures == ()


def test_static_calculate_feature_run_group_passes() -> None:
    result = check_feature_group_shape(make_static_run_group(), input_data=_run_data(), adapter=PythonDictAdapter())
    assert result.failures == (), result.failures


def test_pandas_na_output_passes_determinism() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    result = check_feature_group_shape(make_run_group("emits_na"), input_data=_run_data(), adapter=PandasAdapter())
    assert result.failures == (), result.failures


@pytest.mark.parametrize(
    ("kind", "adapter_cls"),
    [
        ("int64_on_zero_rows", PandasAdapter),
        ("object_date", PandasAdapter),
        ("object_string", PandasAdapter),
        ("tz_out", PyArrowAdapter),
    ],
)
def test_exact_type_run_group_passes(kind: str, adapter_cls: type[Any]) -> None:
    _require_backend(adapter_cls)
    result = check_feature_group_shape(make_run_group(kind), input_data=_run_data(), adapter=adapter_cls())
    assert result.failures == (), result.failures


@pytest.mark.parametrize("adapter_cls", [PyArrowAdapter, PandasAdapter, PolarsAdapter])
def test_default_run_group_passes_with_nullable_int_input(adapter_cls: type[Any]) -> None:
    _require_backend(adapter_cls)
    data = TypedTestData(
        {
            "rg_src": (DataType.INT64, [1, None, 3]),
            "rg_label": (DataType.STRING, ["alpha-secret", "beta-secret", "gamma-secret"]),
        }
    )
    result = check_feature_group_shape(make_run_group(), input_data=data, adapter=adapter_cls())
    assert result.failures == (), result.failures


def test_pandas_object_date_run_notes_value_only_check() -> None:
    _require_backend(PandasAdapter)
    result = check_feature_group_shape(make_run_group("object_date"), input_data=_run_data(), adapter=PandasAdapter())
    assert any("checked by value only" in n for n in result.notes), result.notes


def test_root_group_run_notes_skipped_checks(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    monkeypatch.chdir(tmp_path)
    result = check_feature_group_shape(ListDirectoryFeatureGroup, input_data=TypedTestData({}), adapter=PandasAdapter())
    assert any("zero_rows" in n and "error_values" in n and "not run" in n for n in result.notes), result.notes


def test_assert_forwards_run_keywords() -> None:
    assert_feature_group_shape(make_toy_group(), input_data=_toy_data(), adapter=PythonDictAdapter())
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_feature_group_shape(make_run_group("drops_output"), input_data=_run_data(), adapter=PythonDictAdapter())
    assert "outputs:" in str(excinfo.value)


def test_no_input_data_notes_run_checks_not_run() -> None:
    result = check_feature_group_shape(make_toy_group())
    assert result.passed
    assert any("not run" in n for n in result.notes), result.notes


def test_run_group_conforms() -> None:
    result = check_feature_group_shape(make_run_group(), input_data=_run_data(), adapter=PythonDictAdapter())
    assert result.failures == (), result.failures


def test_missing_declared_input_fails_outputs_and_skips_run_checks() -> None:
    data = TypedTestData({"rg_src": (DataType.INT64, [1])})
    result = check_feature_group_shape(make_run_group(), input_data=data, adapter=PythonDictAdapter())
    assert result.failures
    assert all(f.startswith("outputs:") for f in result.failures), result.failures
    assert any("rg_label" in f for f in result.failures), result.failures


def test_requested_name_already_in_input_data_fails_outputs() -> None:
    data = TypedTestData(
        {
            "rg_src": (DataType.INT64, [1]),
            "rg_label": (DataType.STRING, ["alpha-secret"]),
            "rg_out": (DataType.INT64, [9]),
        }
    )
    result = check_feature_group_shape(make_run_group(), input_data=data, adapter=PythonDictAdapter())
    assert result.failures
    assert all(f.startswith("outputs:") for f in result.failures), result.failures


@pytest.mark.parametrize(
    ("kind", "prefix", "adapter_cls"),
    [
        ("drops_output", "outputs:", PythonDictAdapter),
        ("overwrites_input", "outputs:", PythonDictAdapter),
        ("wrong_type", "type:", PythonDictAdapter),
        ("wrong_type", "type:", PyArrowAdapter),
        ("wrong_type", "type:", PandasAdapter),
        ("narrow_type", "type:", PyArrowAdapter),
        ("narrow_type", "type:", PandasAdapter),
        ("narrow_type", "type:", PolarsAdapter),
        ("loses_schema_on_zero_rows", "zero_rows:", PyArrowAdapter),
        ("list_null_on_zero_rows", "zero_rows:", PolarsAdapter),
        ("iceberg_scan", "zero_rows:", IcebergAdapter),
        ("leaks_cell_in_error", "error_values:", PythonDictAdapter),
        ("nondeterministic", "determinism:", PythonDictAdapter),
        ("nondeterministic_part", "determinism:", PythonDictAdapter),
        ("part_in_one_run", "determinism:", PythonDictAdapter),
        ("plain_in_one_run", "determinism:", PythonDictAdapter),
    ],
)
def test_broken_run_group_fails_only_its_sub_check(kind: str, prefix: str, adapter_cls: type[Any]) -> None:
    _require_backend(adapter_cls)
    adapter = adapter_cls()
    if not adapter.is_available():
        pytest.skip(f"{adapter_cls.__name__} backend unavailable")
    with adapter:
        result = check_feature_group_shape(make_run_group(kind), input_data=_run_data(), adapter=adapter)
    assert not result.passed
    assert all(f.startswith(prefix) for f in result.failures), result.failures


def test_list_directory_runs_on_pandas(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    (tmp_path / "a.txt").write_text("x")
    monkeypatch.chdir(tmp_path)
    result = check_feature_group_shape(ListDirectoryFeatureGroup, input_data=TypedTestData({}), adapter=PandasAdapter())
    assert result.failures == (), result.failures


def test_installed_packages_runs_on_pandas() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    result = check_feature_group_shape(
        InstalledPackagesFeatureGroup, input_data=TypedTestData({}), adapter=PandasAdapter()
    )
    assert result.failures == (), result.failures


def _source_input_feature_case() -> tuple[type[FeatureGroup], Options, str, bool]:
    base = Options(group={DefaultOptionKeys.in_features: frozenset(["src_feature"])})
    return SourceInputFeature, base, "not applicable", True


def _toy_source_case() -> tuple[type[FeatureGroup], Options, str, bool]:
    pytest.importorskip("pyarrow")
    from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import ToySourceFG

    return ToySourceFG, Options(), "not applicable to source groups", False


@pytest.mark.parametrize("case", [_source_input_feature_case, _toy_source_case], ids=["source_input", "toy_source"])
def test_source_group_run_is_not_applicable(case: Callable[[], tuple[type[FeatureGroup], Options, str, bool]]) -> None:
    group, base, note, must_pass = case()
    result = check_feature_group_shape(
        group, base_options=base, input_data=TypedTestData({}), adapter=PythonDictAdapter()
    )
    if must_pass:
        assert result.passed, result.failures
    assert any(n.startswith("run:") and note in n for n in result.notes), result.notes
    assert not any(f.startswith("outputs:") for f in result.failures), result.failures


def _serve_with(
    make_implementation: Callable[..., type[Implementation[Any]]], group: type[FeatureGroup], kind: str
) -> type[Implementation[Any]]:
    run_group = make_run_group(kind)
    return make_implementation(
        group, PythonDictFramework, lambda _, data, features: run_group.calculate_feature(data, features)
    )


def test_operation_group_conforms_with_serving_implementation(
    make_implementation: Callable[..., type[Implementation[Any]]],
) -> None:
    group = make_operation_group()
    _serve_with(make_implementation, group, "ok")
    result = check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert result.failures == (), result.failures
    assert not any(n.startswith("run:") for n in result.notes), result.notes


@pytest.mark.parametrize(
    ("kind", "prefix"),
    [
        ("drops_output", "outputs:"),
        ("wrong_type", "type:"),
        ("leaks_cell_in_error", "error_values:"),
        ("nondeterministic", "determinism:"),
    ],
)
def test_broken_operation_group_fails_only_its_sub_check(
    make_implementation: Callable[..., type[Implementation[Any]]], kind: str, prefix: str
) -> None:
    group = make_operation_group()
    _serve_with(make_implementation, group, kind)
    result = check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert not result.passed
    assert all(f.startswith(prefix) for f in result.failures), result.failures
    assert not any("found no implementation" in f for f in result.failures), result.failures


@pytest.mark.parametrize("other_framework", [None, PandasDataFrame])
def test_operation_group_without_serving_implementation_notes_run_skipped(
    make_implementation: Callable[..., type[Implementation[Any]]], other_framework: type[ComputeFramework] | None
) -> None:
    group = make_operation_group()
    if other_framework is not None:
        make_implementation(group, other_framework, lambda _, data, features: data)
    result = check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert result.failures == (), result.failures
    assert f"run: no implementation of {group.__name__} serves PythonDictFramework" in result.notes, result.notes


class _UnavailableAdapter(PythonDictAdapter):
    def is_available(self) -> bool:
        return False


def test_framework_excluded_by_rule_notes_run_skipped() -> None:
    group = make_run_group("drops_output", rule={PandasDataFrame})
    result = check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert result.failures == (), result.failures
    assert "run: PythonDictFramework excluded by compute_framework_rule" in result.notes, result.notes


def test_unavailable_adapter_notes_run_skipped() -> None:
    result = check_feature_group_shape(
        make_run_group("drops_output"), input_data=_run_data(), adapter=_UnavailableAdapter()
    )
    assert result.failures == (), result.failures
    assert "run: PythonDictFramework is not available" in result.notes, result.notes


@pytest.mark.parametrize("parts", [None, ("*",), ("a", "b")])
def test_part_writing_run_group_conforms(parts: tuple[str, ...] | None) -> None:
    group = make_run_group("parts_ok", output_parts=parts)
    result = check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert result.failures == (), result.failures


@pytest.mark.parametrize("parts", [None, ("*",)])
def test_run_time_part_count_passes_zero_rows(parts: tuple[str, ...] | None) -> None:
    group = make_run_group("parts_count_dependent", output_parts=parts)
    result = check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert result.failures == (), result.failures


@pytest.mark.parametrize(
    ("kind", "prefix", "phrase"),
    [
        ("parts_count_dependent", "zero_rows:", "rg_out~b"),
        ("parts_never_b", "outputs:", "declared part 'rg_out~b' is missing from the result"),
    ],
)
def test_concrete_declared_parts_fail_only_their_sub_check(kind: str, prefix: str, phrase: str) -> None:
    group = make_run_group(kind, output_parts=("a", "b"))
    result = check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert not result.passed
    assert all(f.startswith(prefix) for f in result.failures), result.failures
    assert any(phrase in f for f in result.failures), result.failures


def test_wrong_typed_part_failure_names_the_column() -> None:
    result = check_feature_group_shape(make_run_group("wrong_type_part"), input_data=_run_data())
    assert any("rg_out~a" in f for f in result.failures), result.failures
    assert all(f.startswith("type:") for f in result.failures), result.failures


def test_declared_name_with_tilde_has_no_parts_of_its_own() -> None:
    group = make_run_group("writes_names", output_parts=("a", "b"), names=frozenset({"rg~a", "rg~b"}))
    result = check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert not any("declared part" in f for f in result.failures), result.failures


def test_plain_column_without_part_columns_satisfies_concrete_parts() -> None:
    group = make_run_group("plain_only", output_parts=("a", "b"))
    result = check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert not any(f.startswith("outputs:") for f in result.failures), result.failures


def test_zero_rows_failure_does_not_echo_unmapped_columns() -> None:
    result = check_feature_group_shape(
        make_run_group("pivot_extra"), input_data=_run_data(), adapter=PythonDictAdapter()
    )
    assert any(f.startswith("zero_rows:") for f in result.failures), result.failures
    assert not any("alpha-secret" in f for f in result.failures), result.failures


def test_input_column_with_a_part_name_cannot_fake_an_output() -> None:
    data = TypedTestData(
        {
            "rg_src": (DataType.INT64, [1, 2, 3]),
            "rg_label": (DataType.STRING, ["alpha-secret", "beta-secret", "gamma-secret"]),
            "rg_out~x": (DataType.STRING, ["x", "y", "z"]),
        }
    )
    result = check_feature_group_shape(make_run_group("drops_output"), input_data=data, adapter=PythonDictAdapter())
    assert any(f.startswith("outputs:") and "rg_out" in f for f in result.failures), result.failures
    assert not any(f.startswith("type:") for f in result.failures), result.failures


def _decline(impl: type[Implementation[Any]], calls: list[tuple[str, str | None]], *, raises: bool = False) -> None:
    def accepts(cls: Any, feature_name: str, options: Options, *, part: str | None = None) -> bool:
        calls.append((feature_name, part))
        if raises:
            raise RuntimeError("accepts exploded")
        return False

    setattr(impl, "accepts", classmethod(accepts))


def _recording(
    make_implementation: Callable[..., type[Implementation[Any]]], group: type[FeatureGroup], ran: list[int]
) -> type[Implementation[Any]]:
    run_group = make_run_group("ok")

    def calculate(_: Any, data: Any, features: Any) -> Any:
        ran.append(1)
        return run_group.calculate_feature(data, features)

    return make_implementation(group, PythonDictFramework, calculate)


VARIANT_RECORD = Divergence(kind=DivergenceKind.VARIANT, cause=Cause.BACKEND, variant="v", reason="variant changed")
SUM = Options(group={"op_kind": "sum"})


@pytest.mark.parametrize("how", ["accepts", "supported"])
def test_declined_name_notes_run_skipped_and_does_not_run(
    make_implementation: Callable[..., type[Implementation[Any]]], how: str
) -> None:
    group = make_operation_group(subtypes=True)
    ran: list[int] = []
    impl = _recording(make_implementation, group, ran)
    if how == "accepts":
        _decline(impl, [])
    else:
        setattr(impl, "supported", frozenset({"median"}))
    result = check_feature_group_shape(group, SUM, input_data=_run_data(), adapter=PythonDictAdapter())
    assert result.failures == (), result.failures
    notes = [n for n in result.notes if n.startswith("run:") and "rg_out" in n]
    assert notes, result.notes
    assert group.__name__ in notes[0] and "PythonDictFramework" in notes[0], notes
    assert ran == []


@pytest.mark.parametrize("require_run", [False, True])
def test_raising_accepts_is_a_run_failure_and_does_not_run(
    make_implementation: Callable[..., type[Implementation[Any]]], require_run: bool
) -> None:
    group = make_operation_group()
    ran: list[int] = []
    _decline(_recording(make_implementation, group, ran), [], raises=True)
    result = check_feature_group_shape(
        group, input_data=_run_data(), adapter=PythonDictAdapter(), require_run=require_run
    )
    assert any(f.startswith("run:") and "accepts exploded" in f for f in result.failures), result.failures
    assert any(group.__name__ in f for f in result.failures), result.failures
    assert ran == []


def test_partial_decline_still_checks_the_other_name(
    make_implementation: Callable[..., type[Implementation[Any]]],
) -> None:
    group = make_operation_group(names=frozenset({"rg_a", "rg_b"}))

    def accepts(cls: Any, feature_name: str, options: Options, *, part: str | None = None) -> bool:
        return feature_name != "rg_a"

    impl = make_implementation(group, PythonDictFramework, lambda _, data, features: data)
    setattr(impl, "accepts", classmethod(accepts))
    result = check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert any(n.startswith("run:") and "rg_a" in n for n in result.notes), result.notes
    assert any(f.startswith("outputs:") and "'rg_b'" in f for f in result.failures), result.failures
    assert not any("'rg_a'" in f for f in result.failures), result.failures


def test_unallowed_implementation_variant_notes_run_skipped_and_allowed_one_runs(
    make_implementation: Callable[..., type[Implementation[Any]]],
) -> None:
    group = make_operation_group(variants=frozenset({"v"}))
    ran: list[int] = []
    setattr(_recording(make_implementation, group, ran), "divergences", (VARIANT_RECORD,))
    blocked = check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert blocked.failures == (), blocked.failures
    assert any(n.startswith("run:") and "allow_variants" in n for n in blocked.notes), blocked.notes
    assert ran == []
    allowing = Options(group={str(DefaultOptionKeys.allow_variants): "v"})
    allowed = check_feature_group_shape(group, allowing, input_data=_run_data(), adapter=PythonDictAdapter())
    assert allowed.failures == (), allowed.failures
    assert not any(n.startswith("run:") for n in allowed.notes), allowed.notes
    assert ran != []


def test_undeclared_part_name_reaches_accepts_whole(
    make_implementation: Callable[..., type[Implementation[Any]]],
) -> None:
    group = make_operation_group(names=frozenset({"rg_out~0"}))
    calls: list[tuple[str, str | None]] = []
    _decline(make_implementation(group, PythonDictFramework, lambda _, data, features: data), calls)
    check_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert ("rg_out~0", None) in calls, calls


def _case_no_implementation(mi: Callable[..., type[Implementation[Any]]]) -> tuple[type[FeatureGroup], Any]:
    return make_operation_group(), PythonDictAdapter()


def _case_route_block(mi: Callable[..., type[Implementation[Any]]]) -> tuple[type[FeatureGroup], Any]:
    group = make_operation_group()
    _decline(_serve_with(mi, group, "ok"), [])
    return group, PythonDictAdapter()


def _case_rule_excluded(mi: Callable[..., type[Implementation[Any]]]) -> tuple[type[FeatureGroup], Any]:
    return make_run_group("drops_output", rule={PandasDataFrame}), PythonDictAdapter()


def _case_unavailable(mi: Callable[..., type[Implementation[Any]]]) -> tuple[type[FeatureGroup], Any]:
    return make_run_group("drops_output"), _UnavailableAdapter()


@pytest.mark.parametrize(
    ("require_run", "case", "fails_when_required", "phrase"),
    [
        pytest.param(False, _case_no_implementation, True, "", id="no_implementation-optional"),
        pytest.param(True, _case_no_implementation, True, "", id="no_implementation-required"),
        pytest.param(False, _case_route_block, True, "", id="route_block-optional"),
        pytest.param(True, _case_route_block, True, "", id="route_block-required"),
        pytest.param(True, _case_rule_excluded, True, "compute_framework_rule", id="rule_excluded-required"),
        pytest.param(True, _case_unavailable, False, "", id="unavailable-required"),
    ],
)
def test_skipped_run_is_a_note_unless_required(
    make_implementation: Callable[..., type[Implementation[Any]]],
    require_run: bool,
    case: Callable[..., tuple[type[FeatureGroup], Any]],
    fails_when_required: bool,
    phrase: str,
) -> None:
    group, adapter = case(make_implementation)
    result = check_feature_group_shape(group, input_data=_run_data(), adapter=adapter, require_run=require_run)
    if require_run and fails_when_required:
        assert any(f.startswith("run:") and phrase in f for f in result.failures), result.failures
    else:
        assert result.failures == (), result.failures
        assert any(n.startswith("run:") for n in result.notes), result.notes


@pytest.mark.parametrize(
    "make",
    [make_operation_group, functools.partial(make_run_group, "drops_output", {PandasDataFrame})],
    ids=["no_implementation", "rule_excluded"],
)
def test_assert_fails_on_skipped_run_unless_run_not_required(make: Callable[[], type[FeatureGroup]]) -> None:
    group = make()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter())
    assert "run:" in str(excinfo.value)
    assert_feature_group_shape(group, input_data=_run_data(), adapter=PythonDictAdapter(), require_run=False)


def test_encoding_feature_group_part_columns_are_found_on_pandas() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
    from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncoding  # noqa: F401

    base = Options(
        group={EncodingFeatureGroup.ENCODER_TYPE: "onehot", DefaultOptionKeys.in_features: frozenset(["cat"])}
    )
    data = TypedTestData({"cat": (DataType.STRING, ["a", "b", "c", "a"])})
    result = check_feature_group_shape(EncodingFeatureGroup, base, input_data=data, adapter=PandasAdapter())
    assert not any(f.startswith(("outputs:", "determinism:")) for f in result.failures), result.failures
    assert not any(n.startswith("run:") for n in result.notes), result.notes
