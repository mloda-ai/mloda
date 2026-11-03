"""check_feature_group_shape: a conforming plain FeatureGroup passes, each broken one fails its own sub-check."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from mloda._testing.framework_adapter import PandasAdapter, PyArrowAdapter, PythonDictAdapter
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda._testing.feature_group_shape import assert_feature_group_shape, check_feature_group_shape
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DefaultOptionKeys
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
    """Add out = src * 2 to a dict or Arrow table frame."""
    if _is_columnar_dict(data):
        data[out] = [v * 2 for v in data[src]]
        return data
    import pyarrow.compute as pc

    return data.append_column(out, pc.multiply(data[src], 2))


def make_run_group(kind: str = "ok") -> type[FeatureGroup]:
    """A group over input rg_src (INT64) and rg_label (STRING) that writes rg_out; kind injects one defect."""
    counter = [0]

    class RunGroup(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"rg_out"}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature("rg_src"), Feature("rg_label")}

        @classmethod
        def return_data_type_rule(cls, feature: Feature) -> DataType | None:
            return DataType.INT64

        @classmethod
        def calculate_feature(cls, data: Any, features: Any) -> Any:
            if kind == "drops_output":
                return data
            if kind == "overwrites_input":
                data = _double_into(data, "rg_out", "rg_src")
                data["rg_src"] = [0 for _ in data["rg_src"]]
                return data
            if kind == "wrong_type":
                data["rg_out"] = [str(v) for v in data["rg_src"]]
                return data
            if kind == "loses_schema_on_zero_rows":
                import pyarrow as pa

                if data.num_rows == 0:
                    return data.append_column("rg_out", pa.nulls(0))
                return _double_into(data, "rg_out", "rg_src")
            if kind == "leaks_cell_in_error":
                labels = data["rg_label"]
                if labels and labels[0] != "alpha-secret":
                    raise ValueError("unexpected label, expected 'alpha-secret'")
                return _double_into(data, "rg_out", "rg_src")
            if kind == "nondeterministic":
                counter[0] += 1
                data["rg_out"] = [counter[0]] * len(data["rg_src"])
                return data
            return _double_into(data, "rg_out", "rg_src")

    return RunGroup


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


def _toy_data() -> TypedTestData:
    return TypedTestData({"toy_src": (DataType.INT64, [1, 2, 3])})


def test_toy_group_passes_run_checks_on_python_dict() -> None:
    result = check_feature_group_shape(make_toy_group(), input_data=_toy_data(), adapter=PythonDictAdapter())
    assert result.failures == ()


def test_toy_group_passes_run_checks_on_pyarrow() -> None:
    pytest.importorskip("pyarrow")
    result = check_feature_group_shape(make_toy_group(), input_data=_toy_data(), adapter=PyArrowAdapter())
    assert result.failures == ()


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
        ("loses_schema_on_zero_rows", "zero_rows:", PyArrowAdapter),
        ("leaks_cell_in_error", "error_values:", PythonDictAdapter),
        ("nondeterministic", "determinism:", PythonDictAdapter),
    ],
)
def test_broken_run_group_fails_only_its_sub_check(kind: str, prefix: str, adapter_cls: type[Any]) -> None:
    if adapter_cls is PyArrowAdapter:
        pytest.importorskip("pyarrow")
    result = check_feature_group_shape(make_run_group(kind), input_data=_run_data(), adapter=adapter_cls())
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


def test_source_input_feature_run_is_not_applicable() -> None:
    base = Options(group={DefaultOptionKeys.in_features: frozenset(["src_feature"])})
    result = check_feature_group_shape(
        SourceInputFeature, base_options=base, input_data=TypedTestData({}), adapter=PythonDictAdapter()
    )
    assert result.passed, result.failures
    assert any("not applicable" in n for n in result.notes), result.notes
