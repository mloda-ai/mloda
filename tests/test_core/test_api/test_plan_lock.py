"""Lock-file I/O for write_plan_lock and check_plan_lock, driven by hand-built PlanSteps."""

import dataclasses
import json
from pathlib import Path
from typing import cast

import pytest

from mloda.core.api.plan_lock import _lock_text
from mloda.steward import PlanLockMismatchError, PlanStep, check_plan_lock, write_plan_lock
from mloda.provider import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup


class _MainLike:
    """Plain class posing as one defined in a script."""


_MainLike.__module__ = "__main__"
_MAIN_GROUP = cast("type[FeatureGroup]", _MainLike)


def _compute_step(**overrides: object) -> PlanStep:
    step = PlanStep(
        step_kind="compute",
        feature_names=("lock_io_value",),
        feature_group=PandasAggregatedFeatureGroup,
        compute_framework=PandasDataFrame,
        source_feature_group=None,
        source_compute_framework=None,
        requested_feature_names=("lock_io_value",),
    )
    return dataclasses.replace(step, **overrides)  # type: ignore[arg-type]


def _join_step(**overrides: object) -> PlanStep:
    step = PlanStep(
        step_kind="join",
        feature_names=(),
        feature_group=PandasAggregatedFeatureGroup,
        compute_framework=PandasDataFrame,
        source_feature_group=PandasAggregatedFeatureGroup,
        source_compute_framework=PyArrowTable,
        join_type="inner",
        join_destination_side="left",
    )
    return dataclasses.replace(step, **overrides)  # type: ignore[arg-type]


def _transform_step() -> PlanStep:
    return PlanStep(
        step_kind="transform",
        feature_names=(),
        feature_group=PandasAggregatedFeatureGroup,
        compute_framework=PandasDataFrame,
        source_feature_group=PandasAggregatedFeatureGroup,
        source_compute_framework=PyArrowTable,
    )


def _plan() -> list[PlanStep]:
    return [_compute_step(), _join_step(), _transform_step()]


def test_missing_file_raises_with_path_hint_and_content_and_creates_nothing(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"
    plan = _plan()

    with pytest.raises(PlanLockMismatchError) as excinfo:
        check_plan_lock(plan, lock)

    message = str(excinfo.value)
    assert str(lock) in message
    assert "write_plan_lock" in message
    assert _lock_text(plan) in message
    assert not lock.exists()


def test_written_text_is_canonical_json_and_passes_check(tmp_path: Path) -> None:
    lock = tmp_path / "a" / "b" / "plan.lock"
    plan = _plan()

    write_plan_lock(plan, lock)

    assert b"\r" not in lock.read_bytes()
    text = lock.read_text(encoding="utf-8")
    content = json.loads(text)
    assert text == json.dumps(content, sort_keys=True, indent=2) + "\n"
    assert content["format"] == 1
    assert set(content) == {"format", "requested_features", "compute", "joins", "transforms"}
    assert content["requested_features"] == ["lock_io_value"]
    assert text == _lock_text(plan)
    check_plan_lock(plan, lock)


def test_a_lock_with_an_edited_format_version_fails_check(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"
    plan = _plan()
    write_plan_lock(plan, lock)
    content = json.loads(lock.read_text(encoding="utf-8"))
    content["format"] = 2
    lock.write_text(json.dumps(content, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(PlanLockMismatchError, match="format"):
        check_plan_lock(plan, lock)


def test_a_differing_plan_raises_with_a_unified_diff_and_leaves_the_file_alone(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"
    write_plan_lock(_plan(), lock)
    before = lock.read_text(encoding="utf-8")

    with pytest.raises(PlanLockMismatchError) as excinfo:
        check_plan_lock([_compute_step(compute_framework=PyArrowTable)], lock)

    lines = str(excinfo.value).splitlines()
    assert any(line.startswith("-") and "PandasDataFrame" in line for line in lines)
    assert any(line.startswith("+") and "PyArrowTable" in line for line in lines)
    assert lock.read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param([_compute_step(feature_group=_MAIN_GROUP)], id="compute_feature_group"),
        pytest.param([_join_step(source_feature_group=_MAIN_GROUP)], id="join_source_feature_group"),
    ],
)
def test_write_refuses_main_module_classes_and_writes_nothing(tmp_path: Path, plan: list[PlanStep]) -> None:
    lock = tmp_path / "sub" / "plan.lock"

    with pytest.raises(ValueError, match="__main__"):
        write_plan_lock(plan, lock)

    assert not lock.exists()


def test_a_crlf_rewritten_lock_still_passes_check(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"
    plan = _plan()
    write_plan_lock(plan, lock)
    lock.write_bytes(lock.read_bytes().replace(b"\n", b"\r\n"))

    check_plan_lock(plan, lock)


@pytest.mark.parametrize(
    "second_overrides, distinct",
    [
        pytest.param({}, False, id="identical"),
        pytest.param({"compute_framework": PyArrowTable}, True, id="framework_only"),
    ],
)
def test_compute_steps_give_two_records(second_overrides: dict[str, object], distinct: bool) -> None:
    content = json.loads(_lock_text([_compute_step(), _compute_step(**second_overrides)]))

    assert len(content["compute"]) == 2
    assert (content["compute"][0] != content["compute"][1]) is distinct
