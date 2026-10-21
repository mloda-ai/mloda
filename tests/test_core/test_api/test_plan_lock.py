"""Lock-file I/O for write_plan_lock and check_plan_lock, driven by hand-built PlanSteps."""

import collections
import dataclasses
import hashlib
import importlib.util
import json
import sys
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
from mloda.provider import BaseFeatureGroupVersion, ComputeFramework, FeatureGroup, Implementation
import importlib.metadata
from collections.abc import Iterator

from mloda.core.abstract_plugins.components import base_feature_group_version
from mloda.core.abstract_plugins.components.code_closure import closure_parts, dependency_entry
from mloda.user import Feature, Options, PluginCollector, mloda
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
    neutral_csv_group,
    toy_dac,
)
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_worker_implementation import ToyWorkerFG
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_worker_implementation_impl import (
    ToyWorkerPythonDict,
)
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import ToyFormatBase
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup


class _MainLike:
    """Plain class posing as one defined in a script."""


_MainLike.__module__ = "__main__"


def _inert_group() -> type[ToyFormatBase]:
    """A fresh format group with no claim routes; one per test, so a leaked implementation cannot clash."""

    class _LockInertFG(ToyFormatBase):
        pass

    return _LockInertFG


_MAIN_GROUP = cast("type[FeatureGroup]", _MainLike)


def _compute_step(**overrides: object) -> PlanStep:
    step = PlanStep(
        step_kind="compute",
        feature_names=("lock_io_value",),
        feature_group=AggregatedFeatureGroup,
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
        feature_group=AggregatedFeatureGroup,
        compute_framework=PandasDataFrame,
        source_feature_group=AggregatedFeatureGroup,
        source_compute_framework=PyArrowTable,
        join_type="inner",
        join_destination_side="left",
    )
    return dataclasses.replace(step, **overrides)  # type: ignore[arg-type]


def _transform_step() -> PlanStep:
    return PlanStep(
        step_kind="transform",
        feature_names=(),
        feature_group=AggregatedFeatureGroup,
        compute_framework=PandasDataFrame,
        source_feature_group=AggregatedFeatureGroup,
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
    assert content["format"] == PLAN_LOCK_FORMAT == 7
    assert set(content) == {"format", "requested_features", "compute", "joins", "transforms"}
    assert content["requested_features"] == ["lock_io_value"]
    assert set(content["compute"][0]) == {
        "feature_names",
        "feature_group",
        "compute_framework",
        "compute_framework_reason",
        "specialized_from",
        "result_framework",
        "input_feature_edges",
        "implementation",
        "implementation_hash",
        "implementation_specialized_from",
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


def _impl_path(impl: type[Implementation]) -> str:
    return f"{impl.__module__}:{impl.__qualname__}"


def _impl_step(impl: type[Implementation], implementation_hash: str | None = "h1", **overrides: object) -> PlanStep:
    return _compute_step(implementation=impl, implementation_hash=implementation_hash, **overrides)


def test_the_implementation_fields_are_recorded_and_each_change_fails_check(
    tmp_path: Path, make_implementation: Callable[..., type[Implementation]]
) -> None:
    group = _inert_group()
    first = make_implementation(group, PythonDictFramework, name="LockImplB")
    second = make_implementation(group, PythonDictFramework, base=first, name="LockImplA")
    lock = tmp_path / "plan.lock"
    write_plan_lock([_impl_step(second, implementation_specialized_from=(first, second))], lock)

    record = json.loads(lock.read_text(encoding="utf-8"))["compute"][0]
    assert record["implementation"] == _impl_path(second)
    assert record["implementation_hash"] == "h1"
    assert record["implementation_specialized_from"] == sorted([_impl_path(first), _impl_path(second)])
    for changed in (
        _impl_step(first, implementation_specialized_from=(first, second)),
        _impl_step(second, "h2", implementation_specialized_from=(first, second)),
        _impl_step(second, implementation_specialized_from=(first,)),
    ):
        with pytest.raises(PlanLockMismatchError):
            check_plan_lock([changed], lock)


def test_a_step_without_an_implementation_records_nulls(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"
    write_plan_lock([_compute_step(implementation_hash="ignored")], lock)

    record = json.loads(lock.read_text(encoding="utf-8"))["compute"][0]
    assert record["implementation"] is None
    assert record["implementation_hash"] is None
    assert record["implementation_specialized_from"] == []
    assert "feature_group_hash" not in record


def _explain_compute_step(group: type[FeatureGroup]) -> PlanStep:
    steps = mloda.explain(
        ["toyfmt_ld"],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({group}),
        data_access_collection=toy_dac(h1={"toyfmt_ld": [1, 2]}),
    )
    compute = [step for step in steps if step.step_kind == "compute"]
    assert len(compute) == 1
    return compute[0]


def _dependency_free_hash(impl: type[Implementation]) -> str:
    parts = closure_parts(impl, include_dependencies=False)
    return hashlib.sha256("\n".join(sorted(parts)).encode("utf-8")).hexdigest()


@pytest.mark.parametrize("served", [True, False])
def test_explain_writes_the_serving_implementation_and_its_dependency_free_hash_or_neutral(
    tmp_path: Path, served: bool
) -> None:
    group = ToyWorkerFG if served else neutral_csv_group(tmp_path / "n.csv")
    lock = tmp_path / "plan.lock"
    steps = [_explain_compute_step(group)]

    write_plan_lock(steps, lock)

    record = json.loads(lock.read_text(encoding="utf-8"))["compute"][0]
    if served:
        assert record["implementation"] == _impl_path(ToyWorkerPythonDict)
        assert record["implementation_hash"] == _dependency_free_hash(ToyWorkerPythonDict)
    else:
        assert record["implementation"] == "neutral"
        assert record["implementation_hash"] is None
    check_plan_lock(steps, lock)


def test_an_operation_group_step_round_trips_its_serving_implementation(tmp_path: Path) -> None:
    toy = importlib.import_module(
        "tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_operation_group"
    )
    lock = tmp_path / "plan.lock"
    steps = mloda.explain(
        [Feature(toy.OPTOY_FEATURE, {toy.OPTOY_KEY: "sum"}, compute_framework="PyArrowTable")],
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups({toy.OpToyFG}),
    )

    write_plan_lock(steps, lock)

    record = json.loads(lock.read_text(encoding="utf-8"))["compute"][0]
    assert record["implementation"] == _impl_path(toy.OpToyPyArrow)
    assert record["implementation_hash"] == _dependency_free_hash(toy.OpToyPyArrow)
    assert "feature_group_hash" not in record
    check_plan_lock(steps, lock)


def test_a_subclass_group_step_round_trips_its_own_group_hash(tmp_path: Path) -> None:
    toy = importlib.import_module(
        "tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_operation_group"
    )
    lock = tmp_path / "plan.lock"
    steps = mloda.explain(
        [Feature(toy.OPTOY_CHILD_FEATURE, {toy.OPTOY_KEY: "sum"}, compute_framework="PyArrowTable")],
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups({toy.OpToyChildFG}),
    )

    write_plan_lock(steps, lock)

    record = json.loads(lock.read_text(encoding="utf-8"))["compute"][0]
    assert record["feature_group_hash"] == BaseFeatureGroupVersion.first_party_hash(toy.OpToyChildFG)
    check_plan_lock(steps, lock)


def test_an_unavailable_group_hash_is_written_as_is(tmp_path: Path) -> None:
    lock = tmp_path / "plan.lock"
    steps = [_compute_step(feature_group_hash="unavailable")]

    write_plan_lock(steps, lock)

    assert json.loads(lock.read_text(encoding="utf-8"))["compute"][0]["feature_group_hash"] == "unavailable"
    check_plan_lock(steps, lock)


@pytest.fixture
def pyarrow_version_bump(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], None]]:
    """Returns bump(version): report that version for pyarrow and drop every cached dependency and hash entry."""
    real_version = importlib.metadata.version

    def reset() -> None:
        dependency_entry.cache_clear()
        base_feature_group_version._implementation_hash_cache.clear()

    def bump(version: str) -> None:
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: version if name == "pyarrow" else real_version(name),
        )
        reset()

    yield bump
    monkeypatch.undo()
    reset()


def test_a_dependency_version_change_moves_the_implementation_version_but_not_the_lock_hash(
    tmp_path: Path, pyarrow_version_bump: Callable[[str], None]
) -> None:
    lock = tmp_path / "plan.lock"
    before = _explain_compute_step(ToyWorkerFG)
    write_plan_lock([before], lock)

    pyarrow_version_bump("0.0.1-bumped")
    after = _explain_compute_step(ToyWorkerFG)

    assert before.implementation_version != after.implementation_version
    assert before.implementation_hash == after.implementation_hash == _dependency_free_hash(ToyWorkerPythonDict)
    check_plan_lock([after], lock)


_IMPL_MODULE = "lock_edit_impl_module"
_IMPL_SRC = """\
from mloda.core.abstract_plugins.implementation import Implementation
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import ToyFormatBase


class LockEditGroup(ToyFormatBase):
    pass


class LockEditSubGroup(LockEditGroup):
    unused_value = {sub_value}


def helper() -> int:
    return {value}


class LockEditImpl(Implementation):
    feature_group = LockEditGroup
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(cls, serving, match, features):
        return helper()
"""


def _load_impl(
    directory: Path, value: int, monkeypatch: pytest.MonkeyPatch, sub_value: int | None = None
) -> type[Implementation]:
    """Load LockEditImpl from a file in directory under one fixed module name, so only the body differs."""
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / f"{_IMPL_MODULE}.py"
    source.write_text(
        _IMPL_SRC.format(value=value, sub_value=value if sub_value is None else sub_value), encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location(_IMPL_MODULE, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _IMPL_MODULE, module)
    spec.loader.exec_module(module)
    return cast("type[Implementation]", module.LockEditImpl)


def test_an_edited_subclass_group_body_fails_check_while_the_implementation_hash_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def step_for(directory: Path, value: int) -> PlanStep:
        impl = _load_impl(directory, 1, monkeypatch, sub_value=value)
        sub = cast("type[FeatureGroup]", sys.modules[_IMPL_MODULE].LockEditSubGroup)
        return _impl_step(
            impl,
            BaseFeatureGroupVersion.first_party_hash(impl),
            feature_group=sub,
            feature_group_hash=BaseFeatureGroupVersion.first_party_hash(sub),
        )

    before = step_for(tmp_path / "before", 1)
    lock = tmp_path / "plan.lock"
    write_plan_lock([before], lock)
    check_plan_lock([before], lock)

    after = step_for(tmp_path / "after", 2)

    assert after.implementation_hash == before.implementation_hash
    assert after.feature_group_hash != before.feature_group_hash
    with pytest.raises(PlanLockMismatchError, match="feature_group_hash"):
        check_plan_lock([after], lock)


def test_an_edited_implementation_body_fails_check_while_the_class_path_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _load_impl(tmp_path / "before", 1, monkeypatch)
    before_hash = BaseFeatureGroupVersion.implementation_hash(before)
    before_path = _impl_path(before)
    lock = tmp_path / "plan.lock"
    write_plan_lock([_impl_step(before, before_hash)], lock)
    check_plan_lock([_impl_step(before, before_hash)], lock)
    assert json.loads(lock.read_text(encoding="utf-8"))["compute"][0]["implementation_hash"] == before_hash

    after = _load_impl(tmp_path / "after", 2, monkeypatch)
    after_hash = BaseFeatureGroupVersion.implementation_hash(after)

    assert _impl_path(after) == before_path
    assert after_hash != before_hash
    with pytest.raises(PlanLockMismatchError, match="implementation_hash"):
        check_plan_lock([_impl_step(after, after_hash)], lock)


@pytest.mark.parametrize("where", ["implementation", "implementation_specialized_from"])
def test_write_refuses_a_main_module_implementation_and_writes_nothing(
    tmp_path: Path, where: str, make_implementation: Callable[..., type[Implementation]]
) -> None:
    lock = tmp_path / "plan.lock"
    main_impl = cast("type[Implementation]", _MainLike)
    impl = make_implementation(_inert_group(), PythonDictFramework, name="LockMainImpl")
    step = (
        _impl_step(main_impl)
        if where == "implementation"
        else _impl_step(impl, implementation_specialized_from=(main_impl,))
    )

    with pytest.raises(ValueError, match="__main__"):
        write_plan_lock([step], lock)

    assert not lock.exists()


def test_write_refuses_an_unavailable_implementation_hash_naming_the_class(
    tmp_path: Path, make_implementation: Callable[..., type[Implementation]]
) -> None:
    lock = tmp_path / "plan.lock"
    impl = make_implementation(_inert_group(), PythonDictFramework, name="LockUnhashableImpl")

    with pytest.raises(ValueError, match="unavailable") as excinfo:
        write_plan_lock([_impl_step(impl, "unavailable")], lock)

    assert "LockUnhashableImpl" in str(excinfo.value)
    assert not lock.exists()


def test_plan_hashes_change_with_the_implementation_fields(
    make_implementation: Callable[..., type[Implementation]],
) -> None:
    group = _inert_group()
    first = make_implementation(group, PythonDictFramework, name="LockHashImplA")
    second = make_implementation(group, PythonDictFramework, base=first, name="LockHashImplB")
    base = [_impl_step(first, feature_group_hash="g1")]

    for changed in (
        [_impl_step(second, feature_group_hash="g1")],
        [_impl_step(first, "h2", feature_group_hash="g1")],
        [_impl_step(first, implementation_specialized_from=(second,), feature_group_hash="g1")],
        [_impl_step(first, feature_group_hash="g2")],
    ):
        assert plan_structure_hash(changed) != plan_structure_hash(base)
        assert plan_content_hash(changed) != plan_content_hash(base)


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
    content["format"] = PLAN_LOCK_FORMAT - 1
    assert content["format"] == 6
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


def test_plan_structure_hash_is_the_sha256_of_the_lock_text() -> None:
    plan = _plan()
    digest = plan_structure_hash(plan)

    assert digest == hashlib.sha256(_lock_text(plan).encode("utf-8")).hexdigest()
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_plan_structure_hash_ignores_non_lock_fields() -> None:
    base = [_compute_step(), _join_step()]
    varied = [
        dataclasses.replace(
            base[0],
            feature_set_options=Options({"k": "v"}),
            step_uuid=uuid.uuid4(),
            implementation_version="v1",
            feature_group_version="g1",
        ),
        dataclasses.replace(base[1], join_token=uuid.uuid4()),
    ]
    assert varied[0].feature_set_options is not None

    assert plan_structure_hash(varied) == plan_structure_hash(base)


def _content_plan() -> list[PlanStep]:
    return [
        _compute_step(
            feature_set_options=Options(group={"window": 3}),
            input_feature_edges={"lock_io_value": ("a",)},
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
    # Pinned: the content hash includes the implementation fields, so adding them re-pinned this value; any later
    # change to the hashed content must re-pin it deliberately.
    assert digest == "62f9a6790421ab2966b13bf8f47acabed346aa22b4910f7ca7eb5820f3bd4c63"


def test_plan_content_hash_ignores_run_ids_and_reason_text() -> None:
    base = _content_plan()
    varied = [
        dataclasses.replace(
            base[0],
            step_uuid=uuid.uuid4(),
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
