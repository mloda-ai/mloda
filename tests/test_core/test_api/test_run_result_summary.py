"""Tests for ``RunResult.summary``: the frames and the resolved plan as plain text.

Contract under test: the format ``mloda demo`` prints (``Result`` tables, then ``Plan`` lines) minus its
``Next`` section. Mapping frames render as aligned tables truncated to ``max_rows``, other frames via
``str``, the plan as numbered, aligned step lines; the string has no trailing newline.
"""

import inspect
import subprocess  # nosec B404
import sys
from typing import Any

import pandas as pd
import pytest

# Aliased: a bare ``import mloda.user`` would bind ``mloda`` to the package and collide with the mlodaAPI alias.
import mloda.user as mloda_user
from mloda.core.abstract_plugins.components.input_data.api_data_fg import ApiDataFG
from mloda.interfaces.cli.demo import COLLECTOR, REQUEST
from mloda.user import PlanStep, RunResult, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.python_dict import PythonDictAggregation
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from mloda_plugins.feature_group.input_data.file_formats.json_fg import JsonFG

# Imported for their registration side effect: the README request names the framework by string.
_REGISTERED = (PythonDictFramework, PythonDictAggregation)

README_SUMMARY = """\
Result
customer_id  income
C001         50000
C002         75000
C003         None
C004         60000
C005         85000

age__avg_aggr  income__sum_aggr
37.5           270000
37.5           270000
37.5           270000
37.5           270000
37.5           270000

Plan
step 1  ApiDataFG               age, customer_id, income  (source SampleData)
step 2  AggregatedFeatureGroup  age__avg_aggr, income__sum_aggr  (from age, income)"""


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


class TestMappingFrames:
    """A dict-of-lists frame renders as an aligned table."""

    def test_none_renders_as_none_and_columns_align(self) -> None:
        results = _hand_built([{"a": [1, None], "bb": ["x", "yy"]}])

        assert results.summary() == "Result\na     bb\n1     x\nNone  yy\n\nPlan"

    def test_frames_are_separated_by_one_blank_line_in_list_order(self) -> None:
        results = _hand_built([{"a": [1]}, _Frame()])

        assert results.summary() == "Result\na\n1\n\n<frame as text>\n\nPlan"

    def test_rows_beyond_the_default_max_rows_collapse_into_a_remainder_line(self) -> None:
        results = _hand_built([{"n": list(range(25))}])

        expected = "Result\nn\n" + "\n".join(str(i) for i in range(20)) + "\n... and 5 more rows\n\nPlan"
        assert results.summary() == expected

    def test_max_rows_covering_every_row_prints_them_all_without_a_remainder_line(self) -> None:
        results = _hand_built([{"n": list(range(25))}])

        expected = "Result\nn\n" + "\n".join(str(i) for i in range(25)) + "\n\nPlan"
        assert results.summary(max_rows=25) == expected

    def test_max_rows_of_zero_is_rejected(self) -> None:
        results = _hand_built([{"n": [1]}])

        with pytest.raises(ValueError, match="max_rows"):
            results.summary(max_rows=0)

    def test_negative_max_rows_is_rejected(self) -> None:
        results = _hand_built([{"n": [1]}])

        with pytest.raises(ValueError, match="max_rows"):
            results.summary(max_rows=-1)


class TestNonMappingFrames:
    """Anything that is not a mapping renders as str(frame), without importing pandas."""

    def test_a_non_mapping_frame_renders_via_str(self) -> None:
        results = _hand_built([_Frame()])

        assert results.summary() == "Result\n<frame as text>\n\nPlan"

    def test_a_pandas_frame_renders_via_str(self) -> None:
        frame = pd.DataFrame({"a": [1, None]})
        results = _hand_built([frame])

        assert results.summary() == f"Result\n{frame}\n\nPlan"

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
    """Plan steps render numbered from 1, labelled and detailed per step kind."""

    def test_a_root_compute_step_renders_its_sorted_names_without_a_detail_suffix(self) -> None:
        compute = PlanStep("compute", ("b", "a"), ApiDataFG, None, None, None)
        results = _hand_built([{"a": [1]}], [compute])

        assert results.summary() == "Result\na\n1\n\nPlan\nstep 1  ApiDataFG  a, b"

    def test_transform_and_join_steps_are_labelled_by_kind_and_name_their_groups(self) -> None:
        compute = PlanStep(
            "compute", ("income", "customer_id"), CsvFG, None, None, None, _data_access_identity="api_data:sales"
        )
        transform = PlanStep("transform", (), JsonFG, None, CsvFG, None)
        join = PlanStep("join", (), CsvFG, None, JsonFG, None, join_type="inner")
        results = _hand_built([{"a": [1]}], [compute, transform, join])

        assert results.summary() == (
            "Result\na\n1\n\nPlan\n"
            "step 1  CsvFG      customer_id, income  (source sales)\n"
            "step 2  transform  CsvFG -> JsonFG\n"
            "step 3  join       CsvFG + JsonFG  (inner)"
        )

    def test_plan_detail_names_the_api_data_source_or_the_inputs(self) -> None:
        source = PlanStep("compute", ("x",), ApiDataFG, None, None, None, _data_access_identity="api_data:Orders")
        derived = PlanStep(
            "compute", ("x__sum_aggr",), AggregatedFeatureGroup, None, None, None, input_feature_names=("x",)
        )
        results = _hand_built([{"x": [1]}], [source, derived])

        assert results.summary().splitlines()[-2:] == [
            "step 1  ApiDataFG               x  (source Orders)",
            "step 2  AggregatedFeatureGroup  x__sum_aggr  (from x)",
        ]


class TestPublicExport:
    """summary is reachable on the RunResult exported from mloda.user."""

    def test_summary_is_reachable_on_the_exported_run_result_with_max_rows_defaulting_to_twenty(self) -> None:
        summary = getattr(mloda_user.RunResult, "summary", None)

        assert callable(summary), "mloda.user.RunResult has no attribute 'summary'"
        assert inspect.signature(summary).parameters["max_rows"].default == 20
