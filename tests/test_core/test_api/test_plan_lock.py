"""Lock-file I/O for write_plan_lock and check_plan_lock, driven by hand-built PlanSteps."""

import collections
import dataclasses
import hashlib
import json
import types
import uuid
from pathlib import Path
from collections.abc import Callable
from typing import Any, cast

import pytest

from mloda.core.api.plan_lock import PLAN_LOCK_FORMAT, _lock_text
from mloda.steward import (
    PlanLockMismatchError,
    PlanStep,
    check_plan_lock,
    plan_content_hash,
    plan_structure_hash,
    write_plan_lock,
)
from mloda.core.abstract_plugins.components.credential import RegisteredCredential
from mloda.provider import ComputeFramework, FeatureGroup
from mloda.user import Options
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
        "reader",
        "result_framework",
        "input_feature_edges",
    }
    assert set(content["joins"][0]) == {
        "left_feature_group",
        "right_feature_group",
        "join_type",
        "compute_framework",
        "source_compute_framework",
        "destination_side",
        "join_keys",
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


def test_plan_structure_hash_is_the_sha256_of_the_lock_text() -> None:
    plan = _plan()
    digest = plan_structure_hash(plan)

    assert digest == hashlib.sha256(_lock_text(plan).encode("utf-8")).hexdigest()
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_plan_structure_hash_ignores_non_lock_fields() -> None:
    base = [_compute_step(), _join_step()]
    varied = [
        dataclasses.replace(base[0], feature_set_options=Options({"k": "v"}), step_uuid=uuid.uuid4()),
        dataclasses.replace(base[1], join_token=uuid.uuid4()),
    ]
    assert varied[0].feature_set_options is not None

    assert plan_structure_hash(varied) == plan_structure_hash(base)


def _content_plan() -> list[PlanStep]:
    return [
        _compute_step(
            feature_set_options=Options(group={"window": 3}),
            input_feature_edges={"lock_io_value": ("a",)},
            reader_data_access=(cast(Any, int), "orig"),
        ),
        _join_step(join_keys=("l=r",)),
    ]


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param(lambda: _with(0, compute_framework=PyArrowTable), id="compute_framework"),
        pytest.param(lambda: _with(0, input_feature_edges={"lock_io_value": ("b",)}), id="input_feature_edges"),
        pytest.param(lambda: _with(1, join_keys=("l=other",)), id="join_keys"),
    ],
)
def test_plan_structure_hash_changes_with_a_lock_field(tmp_path: Path, changed: Callable[[], list[PlanStep]]) -> None:
    assert plan_structure_hash(changed()) != plan_structure_hash(_content_plan())

    lock = tmp_path / "plan.lock"
    write_plan_lock(_content_plan(), lock)
    with pytest.raises(PlanLockMismatchError):
        check_plan_lock(changed(), lock)


def test_plan_content_hash_is_a_sha256_hex_and_differs_from_the_structure_hash() -> None:
    plan = _content_plan()
    digest = plan_content_hash(plan)

    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)
    assert digest != plan_structure_hash(plan)
    assert digest == plan_content_hash(_content_plan())
    # Pinned: moving wiring and join keys into the lock must not shift the content hash.
    assert digest == "3795217432afaff05f341f680b5eac2047cb3145faed9cd78d114e79ccf7bd2f"


def test_plan_content_hash_ignores_run_ids_reader_access_and_reason_text() -> None:
    base = _content_plan()
    varied = [
        dataclasses.replace(
            base[0],
            step_uuid=uuid.uuid4(),
            reader_data_access=(cast(Any, int), "somewhere"),
            compute_framework_reason="other reason",
        ),
        dataclasses.replace(base[1], join_token=uuid.uuid4()),
    ]

    assert plan_content_hash(varied) == plan_content_hash(base)


def _with(index: int, **overrides: object) -> list[PlanStep]:
    plan = _content_plan()
    plan[index] = dataclasses.replace(plan[index], **overrides)  # type: ignore[arg-type]
    return plan


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param(lambda: _with(0, feature_names=("other",)), id="feature_name"),
        pytest.param(lambda: _with(0, feature_group=PyArrowTable), id="feature_group"),
        pytest.param(lambda: _with(0, compute_framework=PyArrowTable), id="compute_framework"),
        pytest.param(lambda: _with(0, feature_set_options=Options(group={"window": 4})), id="group_option"),
        pytest.param(lambda: _with(0, input_feature_edges={"lock_io_value": ("b",)}), id="input_edges"),
        pytest.param(lambda: _with(0, reader_data_access=(cast(Any, str), "orig")), id="reader_class"),
        pytest.param(lambda: _with(1, join_type="left"), id="join_type"),
        pytest.param(lambda: _with(1, join_keys=("l=other",)), id="join_keys"),
    ],
)
def test_plan_content_hash_changes_with_a_content_field(changed: Callable[[], list[PlanStep]]) -> None:
    assert plan_content_hash(changed()) != plan_content_hash(_content_plan())


_Conns = collections.namedtuple("_Conns", ["items"])
_OtherConns = collections.namedtuple("_OtherConns", ["items"])


class _ConnList(list[Any]):
    pass


def test_plan_content_hash_scrubs_credential_option_values_only() -> None:
    def hashed(**group: object) -> str:
        return plan_content_hash([_compute_step(feature_set_options=Options(group=group))])

    assert hashed(password="a1") == hashed(password="b2")  # nosec B106
    assert hashed(conn={"password": "a1", "host": "h"}) == hashed(conn={"password": "b2", "host": "h"})  # nosec B105
    assert hashed(conn={"api_key": "a1"}) == hashed(conn={"api_key": "b2"})
    assert hashed(window=3) != hashed(window=4)
    assert hashed(conns=[{"password": "a1"}]) == hashed(conns=[{"password": "b2"}])  # nosec B105
    assert hashed(conns=({"password": "a1"},)) == hashed(conns=({"password": "b2"},))  # nosec B105
    assert hashed(c={"l": [{"api_key": "a1"}]}) == hashed(c={"l": [{"api_key": "b2"}]})
    assert hashed(c=types.MappingProxyType({"password": "a1"})) == hashed(  # nosec B105
        c=types.MappingProxyType({"password": "b2"})  # nosec B105
    )
    assert hashed(c=_Conns(items=[{"password": "a1"}])) == hashed(  # nosec B105
        c=_Conns(items=[{"password": "b2"}])  # nosec B105
    )
    assert hashed(c=_ConnList([{"password": "a1"}])) == hashed(  # nosec B105
        c=_ConnList([{"password": "b2"}])  # nosec B105
    )
    assert hashed(c=_Conns(items=[1])) != hashed(c=_OtherConns(items=[1]))
    assert hashed(conn=RegisteredCredential(host="h1", password="p")) == hashed(  # nosec B106
        conn=RegisteredCredential(host="h2", password="p")  # nosec B106
    )
    assert hashed(conns=[{"host": "a"}]) != hashed(conns=[{"host": "b"}])


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("https://h/x.csv?rev=1", "https://h/x.csv?rev=2"),
        ("s3://b/dir/a%20b/one.csv", "s3://b/dir/a%20b/two.csv"),
    ],
)
def test_plan_content_hash_keeps_non_credential_url_values_in_full(left: str, right: str) -> None:
    def hashed(path: str) -> str:
        return plan_content_hash([_compute_step(feature_set_options=Options(group={"path": path}))])

    assert hashed(left) != hashed(right)


def test_plan_content_hash_is_independent_of_set_and_dict_order() -> None:
    first = _compute_step(feature_set_options=Options(group={"s": {"a", "b", "c"}, "d": {"x": 1, "y": 2}}))
    second = _compute_step(feature_set_options=Options(group={"d": {"y": 2, "x": 1}, "s": {"c", "b", "a"}}))

    assert plan_content_hash([first]) == plan_content_hash([second])

    proxy_a = types.MappingProxyType({"x": 1, "y": 2})
    proxy_b = types.MappingProxyType({"y": 2, "x": 1})
    third = _compute_step(feature_set_options=Options(group={"m": proxy_a}))
    fourth = _compute_step(feature_set_options=Options(group={"m": proxy_b}))
    assert plan_content_hash([third]) == plan_content_hash([fourth])
