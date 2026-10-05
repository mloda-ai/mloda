"""Lock-file I/O for write_plan_lock and check_plan_lock, driven by hand-built PlanSteps."""

import dataclasses
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any, cast

import pytest

from mloda.core.api.plan_lock import PLAN_LOCK_FORMAT, _lock_text
from mloda.steward import PlanLockMismatchError, PlanStep, check_plan_lock, write_plan_lock
from mloda.provider import ComputeFramework, FeatureGroup
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
    assert content["format"] == PLAN_LOCK_FORMAT == 4
    assert set(content) == {"format", "requested_features", "compute", "joins", "transforms"}
    assert content["requested_features"] == ["lock_io_value"]
    assert set(content["compute"][0]) == {
        "feature_names",
        "feature_group",
        "compute_framework",
        "compute_framework_reason",
        "specialized_from",
        "result_framework",
    }
    assert set(content["joins"][0]) == {
        "left_feature_group",
        "right_feature_group",
        "join_type",
        "compute_framework",
        "source_compute_framework",
        "destination_side",
    }
    assert set(content["transforms"][0]) == {"feature_group", "from_compute_framework", "to_compute_framework"}
    assert text == _lock_text(plan)
    check_plan_lock(plan, lock)


def test_the_result_framework_is_recorded_per_compute_record_and_a_changed_one_fails_check(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"
    write_plan_lock([_compute_step(result_framework=PyArrowTable)], lock)

    recorded = json.loads(lock.read_text(encoding="utf-8"))["compute"][0]["result_framework"]
    assert recorded == f"{PyArrowTable.__module__}:{PyArrowTable.__qualname__}"
    with pytest.raises(PlanLockMismatchError):
        check_plan_lock([_compute_step(result_framework=PandasDataFrame)], lock)


def test_write_refuses_a_main_module_result_framework_and_writes_nothing(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"

    with pytest.raises(ValueError, match="__main__"):
        write_plan_lock([_compute_step(result_framework=cast("type[ComputeFramework]", _MainLike))], lock)

    assert not lock.exists()


def test_the_compute_reason_is_recorded_and_a_changed_reason_fails_check(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"
    write_plan_lock([_compute_step(compute_framework_reason="pinned")], lock)

    assert json.loads(lock.read_text(encoding="utf-8"))["compute"][0]["compute_framework_reason"] == "pinned"
    with pytest.raises(PlanLockMismatchError):
        check_plan_lock([_compute_step(compute_framework_reason="default order")], lock)


def test_a_lock_with_an_edited_format_version_fails_check(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"
    plan = _plan()
    write_plan_lock(plan, lock)
    content = json.loads(lock.read_text(encoding="utf-8"))
    content["format"] = PLAN_LOCK_FORMAT + 1
    lock.write_text(json.dumps(content, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(PlanLockMismatchError, match="format"):
        check_plan_lock(plan, lock)


def test_a_lock_written_in_the_previous_format_with_a_reader_entry_fails_check(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"
    plan = _plan()
    write_plan_lock(plan, lock)
    content = json.loads(lock.read_text(encoding="utf-8"))
    content["format"] = 3
    for record in content["compute"]:
        record["reader"] = None
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


def test_write_allows_main_prefixed_feature_names_when_classes_are_in_modules(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"
    plan = [_compute_step(feature_names=("__main__:x",), requested_feature_names=("__main__:x",))]

    write_plan_lock(plan, lock)

    check_plan_lock(plan, lock)


@pytest.mark.parametrize("operation", [write_plan_lock, check_plan_lock])
def test_unknown_step_kind_raises_value_error(tmp_path: Path, operation: Callable[..., None]) -> None:
    plan = [dataclasses.replace(_compute_step(), step_kind=cast(Any, "bogus"))]

    with pytest.raises(ValueError):
        operation(plan, tmp_path / "plan.lock")


@pytest.mark.parametrize(
    "rewrite",
    [
        pytest.param(lambda data: data.replace(b"\n", b"\r\n"), id="crlf"),
        pytest.param(lambda data: b"\xef\xbb\xbf" + data, id="bom"),
    ],
)
def test_a_rewritten_lock_still_passes_check(tmp_path: Path, rewrite: Callable[[bytes], bytes]) -> None:
    lock = tmp_path / "plan.lock"
    plan = _plan()
    write_plan_lock(plan, lock)
    lock.write_bytes(rewrite(lock.read_bytes()))

    check_plan_lock(plan, lock)
