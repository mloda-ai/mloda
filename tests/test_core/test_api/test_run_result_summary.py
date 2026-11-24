"""Tests for ``RunResult.summary``: the frames and the resolved plan as plain text.

Contract: the ``Result``/``Plan`` part of ``mloda demo``. Mapping, pandas, PyArrow and eager Polars frames
render as GitHub-flavored Markdown pipe tables truncated to ``max_rows``, other frames via ``str``, the plan
as one ``step | plugin | features | from`` pipe table; the string has no trailing newline.
"""

import inspect
import subprocess  # nosec B404
import sys
from collections.abc import Callable
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

# Aliased: a bare ``import mloda.user`` would bind ``mloda`` to the package and collide with the mlodaAPI alias.
import mloda.user as mloda_user
from mloda.core.abstract_plugins.components.input_data.api_data_fg import ApiDataFG
from mloda.interfaces.cli.demo import COLLECTOR, REQUEST
from mloda.user import PlanStep, RunResult, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregation
from mloda_plugins.feature_group.experimental.aggregated_feature_group.python_dict import PythonDictAggregation
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from mloda_plugins.feature_group.input_data.file_formats.json_fg import JsonFG

# Imported for their registration side effect: the README request names the framework by string.
_REGISTERED = (PythonDictFramework, PythonDictAggregation, PandasDataFrame, PandasAggregation)

README_SUMMARY = """\
Result
| customer_id | income |
|-------------|--------|
| C001        | 50000  |
| C002        | 75000  |
| C003        | None   |
| C004        | 60000  |
| C005        | 85000  |

| age__avg_aggr | income__sum_aggr |
|---------------|------------------|
| 37.5          | 270000           |
| 37.5          | 270000           |
| 37.5          | 270000           |
| 37.5          | 270000           |
| 37.5          | 270000           |

Plan
| step | plugin                 | features                        | from              |
|------|------------------------|---------------------------------|-------------------|
| 1    | ApiDataFG              | age, customer_id, income        | source SampleData |
| 2    | AggregatedFeatureGroup | age__avg_aggr, income__sum_aggr | age, income       |"""

README_PANDAS_RESULT = """\
Result
| customer_id | income  |
|-------------|---------|
| C001        | 50000.0 |
| C002        | 75000.0 |
| C003        | nan     |
| C004        | 60000.0 |
| C005        | 85000.0 |

| age__avg_aggr | income__sum_aggr |
|---------------|------------------|
| 37.5          | 270000.0         |
| 37.5          | 270000.0         |
| 37.5          | 270000.0         |
| 37.5          | 270000.0         |
| 37.5          | 270000.0         |"""

# Each builds a frame of one backend from a dict of columns.
_DATAFRAME_BACKENDS: list[Callable[[dict[str, list[Any]]], Any]] = [pd.DataFrame, pa.table, pl.DataFrame]
_BACKEND_IDS = ["pandas", "pyarrow", "polars"]


class _Frame:
    """A non-mapping frame whose text form is known."""

    def __str__(self) -> str:
        return "<frame as text>"


# Helpers return Any on purpose: ``summary`` is exactly what these tests assert at runtime.
def _run_readme_example() -> Any:
    return mloda.run_all(
        features=REQUEST["features"],
        compute_frameworks=["PythonDictFramework"],
        api_data=REQUEST["data"],
        plugin_collector=COLLECTOR,
    )


def _run_readme_example_on_pandas() -> Any:
    return mloda.run_all(
        features=REQUEST["features"],
        compute_frameworks=["PandasDataFrame"],
        api_data=REQUEST["data"],
        plugin_collector=COLLECTOR,
    )


def _hand_built(frames: list[Any], plan: list[PlanStep] | None = None) -> Any:
    return RunResult(frames, plan or [])


class TestSummaryOfTheReadmeExample:
    """summary() of the README request is the demo's Result and Plan text."""

    def test_summary_renders_the_frames_and_the_plan(self) -> None:
        results = _run_readme_example()

        assert results.summary() == README_SUMMARY

    def test_summary_has_no_trailing_newline(self) -> None:
        results = _run_readme_example()

        assert not results.summary().endswith("\n")

    def test_the_readme_request_on_pandas_renders_its_frames_as_pipe_tables(self) -> None:
        results = _run_readme_example_on_pandas()

        assert results.summary().split("\n\nPlan\n")[0] == README_PANDAS_RESULT


class TestMappingFrames:
    """A dict-of-lists frame renders as a GFM pipe table."""

    def test_none_renders_as_none_and_columns_pad_to_the_widest_cell(self) -> None:
        results = _hand_built([{"a": [1, None], "bb": ["x", "yy"]}])

        assert results.summary() == ("Result\n| a    | bb |\n|------|----|\n| 1    | x  |\n| None | yy |\n\nPlan")

    def test_a_pipe_in_a_header_or_cell_is_escaped_and_counted_in_the_width(self) -> None:
        results = _hand_built([{"a|b": ["x|y", "z"]}])

        assert results.summary() == "Result\n| a\\|b |\n|------|\n| x\\|y |\n| z    |\n\nPlan"

    def test_frames_are_separated_by_one_blank_line_in_list_order(self) -> None:
        results = _hand_built([{"a": [1]}, _Frame()])

        assert results.summary() == "Result\n| a |\n|---|\n| 1 |\n\n<frame as text>\n\nPlan"

    def test_rows_beyond_the_default_max_rows_collapse_into_a_remainder_line(self) -> None:
        results = _hand_built([{"n": list(range(25))}])

        rows = "\n".join(f"| {i:<2} |" for i in range(20))
        expected = f"Result\n| n  |\n|----|\n{rows}\n... and 5 more rows\n\nPlan"
        assert results.summary() == expected

    def test_max_rows_covering_every_row_prints_them_all_without_a_remainder_line(self) -> None:
        results = _hand_built([{"n": list(range(25))}])

        rows = "\n".join(f"| {i:<2} |" for i in range(25))
        expected = f"Result\n| n  |\n|----|\n{rows}\n\nPlan"
        assert results.summary(max_rows=25) == expected

    def test_max_rows_of_zero_is_rejected(self) -> None:
        results = _hand_built([{"n": [1]}])

        with pytest.raises(ValueError, match="max_rows"):
            results.summary(max_rows=0)

    def test_negative_max_rows_is_rejected(self) -> None:
        results = _hand_built([{"n": [1]}])

        with pytest.raises(ValueError, match="max_rows"):
            results.summary(max_rows=-1)


class TestDataFrameFrames:
    """pandas, PyArrow and eager Polars frames render as the pipe table of the same dict-of-lists."""

    def test_a_pandas_frame_renders_its_columns_in_order_with_pandas_values_and_no_index(self) -> None:
        results = _hand_built([pd.DataFrame({"name": ["x", "yy"], "a": [1, None]}, index=[10, 11])])

        assert results.summary() == ("Result\n| name | a   |\n|------|-----|\n| x    | 1.0 |\n| yy   | nan |\n\nPlan")

    def test_a_pyarrow_table_renders_its_columns_in_order_with_null_as_none(self) -> None:
        results = _hand_built([pa.table({"name": ["x", "yy"], "a": [1, None]})])

        assert results.summary() == (
            "Result\n| name | a    |\n|------|------|\n| x    | 1    |\n| yy   | None |\n\nPlan"
        )

    def test_a_polars_frame_renders_its_columns_in_order_with_null_as_none(self) -> None:
        results = _hand_built([pl.DataFrame({"name": ["x", "yy"], "a": [1, None]})])

        assert results.summary() == (
            "Result\n| name | a    |\n|------|------|\n| x    | 1    |\n| yy   | None |\n\nPlan"
        )

    @pytest.mark.parametrize("build", _DATAFRAME_BACKENDS, ids=_BACKEND_IDS)
    def test_rows_beyond_the_default_max_rows_collapse_into_a_remainder_line(
        self, build: Callable[[dict[str, list[Any]]], Any]
    ) -> None:
        results = _hand_built([build({"n": list(range(25))})])

        rows = "\n".join(f"| {i:<2} |" for i in range(20))
        expected = f"Result\n| n  |\n|----|\n{rows}\n... and 5 more rows\n\nPlan"
        assert results.summary() == expected


class TestNonMappingFrames:
    """Anything else renders as str(frame), without importing pandas, polars or pyarrow."""

    def test_a_non_mapping_frame_renders_via_str(self) -> None:
        results = _hand_built([_Frame()])

        assert results.summary() == "Result\n<frame as text>\n\nPlan"

    def test_a_polars_lazy_frame_renders_via_str_without_being_collected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lazy = pl.DataFrame({"a": [1, None]}).lazy()
        expected = f"Result\n{lazy}\n\nPlan"

        def _no_collect(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("summary() collected a LazyFrame")

        monkeypatch.setattr(pl.LazyFrame, "collect", _no_collect)

        assert _hand_built([lazy]).summary() == expected

    @pytest.mark.timeout(60)
    def test_rendering_does_not_import_polars_or_pyarrow(self) -> None:
        probe = (
            "import sys\n"
            "from mloda.core.api.run_result import RunResult\n"
            "names = ('pandas', 'polars', 'pyarrow')\n"
            "print([n in sys.modules for n in names])\n"
            "class Frame:\n"
            "    def __str__(self):\n"
            "        return 'frame'\n"
            "print(RunResult([{'a': [1, None]}, Frame()], []).summary())\n"
            "print([n in sys.modules for n in names])\n"
        )
        # Safe: argv is the running interpreter and a constant script, no shell.
        run = subprocess.run(  # nosec B603
            [sys.executable, "-I", "-c", probe], capture_output=True, text=True, timeout=60
        )

        assert run.returncode == 0, run.stderr
        lines = run.stdout.splitlines()
        assert lines[0] == "[False, False, False]", f"importing run_result loaded a frame library:\n{run.stdout}"
        assert lines[-1] == "[False, False, False]", f"summary() imported a frame library:\n{run.stdout}"

    @pytest.mark.timeout(60)
    def test_rendering_does_not_import_pandas(self) -> None:
        probe = (
            "import sys\n"
            "from mloda.core.api.run_result import RunResult\n"
            "class Frame:\n"
            "    def __str__(self):\n"
            "        return 'frame'\n"
            "print(RunResult([{'a': [1, None]}, Frame()], []).summary())\n"
            "print('pandas' in sys.modules)\n"
        )
        # Safe: argv is the running interpreter and a constant script, no shell.
        run = subprocess.run(  # nosec B603
            [sys.executable, "-I", "-c", probe], capture_output=True, text=True, timeout=60
        )

        assert run.returncode == 0, run.stderr
        assert run.stdout.splitlines()[-1] == "False", f"summary() imported pandas:\n{run.stdout}"


class TestPlanLines:
    """The plan renders as one ``step | plugin | features | from`` pipe table, numbered from 1."""

    def test_a_root_compute_step_renders_its_sorted_names_with_an_empty_from_cell(self) -> None:
        compute = PlanStep("compute", ("b", "a"), ApiDataFG, None, None, None)
        results = _hand_built([{"a": [1]}], [compute])

        assert results.summary() == (
            "Result\n| a |\n|---|\n| 1 |\n\nPlan\n"
            "| step | plugin    | features | from |\n"
            "|------|-----------|----------|------|\n"
            "| 1    | ApiDataFG | a, b     |      |"
        )

    def test_transform_and_join_steps_are_labelled_by_kind_and_name_their_groups(self) -> None:
        compute = PlanStep(
            "compute", ("income", "customer_id"), CsvFG, None, None, None, _data_access_identity="api_data:sales"
        )
        transform = PlanStep("transform", (), JsonFG, None, CsvFG, None)
        join = PlanStep("join", (), CsvFG, None, JsonFG, None, join_type="inner")
        results = _hand_built([{"a": [1]}], [compute, transform, join])

        assert results.summary() == (
            "Result\n| a |\n|---|\n| 1 |\n\nPlan\n"
            "| step | plugin    | features            | from                   |\n"
            "|------|-----------|---------------------|------------------------|\n"
            "| 1    | CsvFG     | customer_id, income | source sales           |\n"
            "| 2    | transform |                     | CsvFG -> JsonFG        |\n"
            "| 3    | join      |                     | CsvFG + JsonFG (inner) |"
        )

    def test_the_from_cell_names_the_api_data_source_or_the_inputs(self) -> None:
        source = PlanStep("compute", ("x",), ApiDataFG, None, None, None, _data_access_identity="api_data:Orders")
        derived = PlanStep(
            "compute", ("x__sum_aggr",), AggregatedFeatureGroup, None, None, None, input_feature_names=("x",)
        )
        results = _hand_built([{"x": [1]}], [source, derived])

        assert results.summary().splitlines()[-2:] == [
            "| 1    | ApiDataFG              | x           | source Orders |",
            "| 2    | AggregatedFeatureGroup | x__sum_aggr | x             |",
        ]

    def test_an_empty_plan_renders_the_heading_without_table_lines(self) -> None:
        results = _hand_built([{"a": [1]}])

        assert results.summary().endswith("| 1 |\n\nPlan")


class TestPublicExport:
    """summary is reachable on the RunResult exported from mloda.user."""

    def test_summary_is_reachable_on_the_exported_run_result_with_max_rows_defaulting_to_twenty(self) -> None:
        summary = getattr(mloda_user.RunResult, "summary", None)

        assert callable(summary), "mloda.user.RunResult has no attribute 'summary'"
        assert inspect.signature(summary).parameters["max_rows"].default == 20
