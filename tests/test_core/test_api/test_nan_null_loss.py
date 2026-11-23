"""Plan-time metadata for potential NaN/null loss at framework transfers (#2964)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, TypeVar, cast, get_args, get_type_hints

import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.api.plan_lock import PLAN_LOCK_FORMAT, _build, _lock_text
from mloda.interfaces.service.planner import _step_record
from mloda.provider import (
    ApiDataFG,
    ClaimRoute,
    ComputeFramework,
    DataCreatorFeatureGroup,
    FeatureGroup,
    FeatureSet,
    NamePolicy,
    SourceMatch,
)
from mloda.steward import (
    PlanLockMismatchError,
    check_plan_lock,
    plan_content_hash,
    plan_structure_hash,
    write_plan_lock,
)
from mloda.user import DataAccessCollection, Feature, FeatureName, Options, PlanStep, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
    TOY_KEY,
    ToySourceBase,
)


SOURCE = "nan_null_2964_source"
CONSUMER = "nan_null_2964_consumer"
NEUTRAL = "toyfmt_nan_null_2964"
SOURCE_FRAMEWORK = "source_framework_2964"
NATIVE_TYPE = "neutral_type_2964"
FRAMEWORKS = (PyArrowTable, PandasDataFrame, SqliteFramework, PythonDictFramework)
StepKind = Literal["compute", "join", "transform"]
_T = TypeVar("_T")


class NanNull2964Root(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {SOURCE}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return set(FRAMEWORKS)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise AssertionError("Explaining transfer metadata must not execute a feature")


class NanNull2964Consumer(FeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {CONSUMER}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return set(FRAMEWORKS)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {Feature(SOURCE, compute_framework=options.get(SOURCE_FRAMEWORK))}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise AssertionError("Explaining transfer metadata must not execute a feature")


class NanNull2964NeutralSource(ToySourceBase):
    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.CHECKED, True),)

    @classmethod
    def neutral_type(cls, match: SourceMatch) -> type | None:
        return cast("type | None", match.access.credentials[NATIVE_TYPE])

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        raise AssertionError("Explaining transfer metadata must not load or scan neutral data")


class NanNull2964SqliteSubtype(SqliteRelation):
    pass


class NanNull2964GenericSqlite(SqliteRelation, Generic[_T]):
    pass


PLUGINS = PluginCollector.enabled_feature_groups({NanNull2964Root, NanNull2964Consumer})
TRANSFER_CASES = [
    pytest.param(PyArrowTable, PandasDataFrame, True, id="arrow-to-pandas"),
    pytest.param(PyArrowTable, SqliteFramework, True, id="arrow-to-sqlite"),
    pytest.param(PythonDictFramework, PandasDataFrame, True, id="dict-to-pandas"),
    pytest.param(PythonDictFramework, SqliteFramework, True, id="dict-to-sqlite"),
    pytest.param(PandasDataFrame, SqliteFramework, True, id="pandas-to-sqlite"),
    pytest.param(SqliteFramework, PandasDataFrame, False, id="sqlite-to-pandas"),
    pytest.param(PandasDataFrame, PyArrowTable, False, id="pandas-to-arrow"),
    pytest.param(SqliteFramework, PyArrowTable, False, id="sqlite-to-arrow"),
    pytest.param(PyArrowTable, PythonDictFramework, False, id="arrow-to-dict"),
    pytest.param(PythonDictFramework, PyArrowTable, False, id="dict-to-arrow"),
]


def _potential_loss(step: PlanStep) -> bool:
    value = getattr(step, "potential_nan_null_loss", False)
    assert type(value) is bool
    return value


def _transfer_plan(
    source: type[ComputeFramework], destination: type[ComputeFramework], data_type: DataType | None = None
) -> list[PlanStep]:
    from contextlib import closing, nullcontext
    import sqlite3

    context = {SOURCE_FRAMEWORK: source.get_class_name()}
    if destination is SqliteFramework:
        context["connection_handle"] = "nan_null_transfer"
    request = Feature(
        CONSUMER,
        Options(context=context),
        compute_framework=destination.get_class_name(),
        data_type=data_type,
    )
    connection_scope = closing(sqlite3.connect(":memory:")) if destination is SqliteFramework else nullcontext(None)
    with connection_scope as connection:
        collection = None if connection is None else DataAccessCollection(connections={"nan_null_transfer": connection})
        return mloda.explain(
            [request],
            compute_frameworks=[source, destination],
            plugin_collector=PLUGINS,
            data_access_collection=collection,
        )


def _neutral_plan(native_type: Any, framework: type[ComputeFramework], data: Any = None) -> list[PlanStep]:
    collection = DataAccessCollection(
        credentials={
            "nan_null_fixture": {
                TOY_KEY: {NEUTRAL: [float("nan"), None, 1.0]},
                NATIVE_TYPE: native_type,
                "neutral_fixture_data": data,
            }
        }
    )
    return mloda.explain(
        [NEUTRAL],
        compute_frameworks=[framework],
        data_access_collection=collection,
        plugin_collector=PluginCollector.enabled_feature_groups({NanNull2964NeutralSource}),
    )


def _manual_step(kind: StepKind, potential_loss: bool | None = None) -> PlanStep:
    step = PlanStep(
        step_kind=kind,
        feature_names=(SOURCE,) if kind == "compute" else (),
        feature_group=NanNull2964Root,
        compute_framework=PandasDataFrame,
        source_feature_group=None if kind == "compute" else NanNull2964Root,
        source_compute_framework=None if kind == "compute" else PyArrowTable,
        requested_feature_names=(SOURCE,) if kind == "compute" else (),
        join_type="inner" if kind == "join" else None,
    )
    if potential_loss is None:
        return step
    extra: dict[str, Any] = {"potential_nan_null_loss": potential_loss}
    return dataclasses.replace(step, **extra)


@pytest.mark.parametrize(
    ("framework", "semantics"),
    [
        (ComputeFramework, "distinct"),
        (PyArrowTable, "distinct"),
        (PythonDictFramework, "distinct"),
        (DuckDBFramework, "distinct"),
        (PandasDataFrame, "dtype_dependent"),
        (SqliteFramework, "collapsed"),
    ],
)
def test_framework_declares_nan_null_semantics_once(framework: type[ComputeFramework], semantics: str) -> None:
    assert getattr(framework, "nan_null_semantics")() == semantics


def test_framework_semantics_return_type_lists_all_three_states() -> None:
    annotation = get_type_hints(getattr(ComputeFramework, "nan_null_semantics"))["return"]

    assert set(get_args(annotation)) == {"distinct", "collapsed", "dtype_dependent"}


def test_plan_step_appends_a_frozen_boolean_with_a_false_default() -> None:
    step = PlanStep("compute", (), None, None, None, None)
    field = dataclasses.fields(PlanStep)[-1]

    assert field.name == "potential_nan_null_loss"
    assert field.default is False
    assert getattr(step, field.name) is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(step, field.name, True)


@pytest.mark.parametrize(("source", "destination", "expected"), TRANSFER_CASES)
def test_explicit_transfer_marks_only_the_actual_transfer_step(
    source: type[ComputeFramework], destination: type[ComputeFramework], expected: bool
) -> None:
    plan = _transfer_plan(source, destination)
    transforms = [step for step in plan if step.step_kind == "transform"]

    assert len(transforms) == 1
    assert transforms[0].source_compute_framework is source
    assert transforms[0].compute_framework is destination
    assert _potential_loss(transforms[0]) is expected
    assert all(not _potential_loss(step) for step in plan if step.step_kind == "compute")
    assert all(step.divergences == () and step.allowed_variants == () for step in plan)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_native_data_without_a_framework_transfer_is_quiet(framework: type[ComputeFramework]) -> None:
    plan = _transfer_plan(framework, framework)

    assert all(step.step_kind == "compute" for step in plan)
    assert all(step.compute_framework is framework for step in plan)
    assert all(not _potential_loss(step) for step in plan)


@pytest.mark.parametrize("requested_type", [None, DataType.INT32, DataType.STRING])
def test_requested_output_type_cannot_prove_the_transferred_frame_has_no_float_data(
    requested_type: DataType | None,
) -> None:
    plan = _transfer_plan(PyArrowTable, PandasDataFrame, requested_type)
    transfer = next(step for step in plan if step.step_kind == "transform")

    assert _potential_loss(transfer) is True


@pytest.mark.parametrize(("source", "destination", "expected"), TRANSFER_CASES)
def test_requested_result_conversion_marks_the_compute_record(
    source: type[ComputeFramework], destination: type[ComputeFramework], expected: bool
) -> None:
    from contextlib import closing, nullcontext
    import sqlite3

    connection_scope = closing(sqlite3.connect(":memory:")) if destination is SqliteFramework else nullcontext(None)
    with connection_scope as connection:
        collection = None if connection is None else DataAccessCollection(connections={"nan_null_output": connection})
        plan = mloda.explain(
            [Feature(SOURCE, compute_framework=source.get_class_name())],
            compute_frameworks=[source],
            output_framework=destination,
            plugin_collector=PLUGINS,
            data_access_collection=collection,
        )

    assert len(plan) == 1
    step = plan[0]
    assert step.step_kind == "compute"
    assert step.compute_framework is source
    assert step.result_framework is destination
    assert step.source_compute_framework is None
    assert _potential_loss(step) is expected


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_explicit_same_framework_result_conversion_is_quiet(framework: type[ComputeFramework]) -> None:
    plan = mloda.explain(
        [Feature(SOURCE, compute_framework=framework.get_class_name())],
        compute_frameworks=[framework],
        output_framework=framework,
        plugin_collector=PLUGINS,
    )

    assert len(plan) == 1
    assert plan[0].result_framework is framework
    assert not _potential_loss(plan[0])


def test_declared_result_framework_is_also_checked_without_an_output_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PyArrowTable, "result_framework", classmethod(lambda cls: PandasDataFrame))
    plan = mloda.explain([SOURCE], compute_frameworks=[PyArrowTable], plugin_collector=PLUGINS)

    assert plan[0].result_framework is PandasDataFrame
    assert _potential_loss(plan[0]) is True


def test_output_conversion_does_not_mark_unrequested_intermediate_results() -> None:
    plan = mloda.explain(
        [
            Feature(
                CONSUMER,
                Options(context={SOURCE_FRAMEWORK: PyArrowTable.get_class_name()}),
                compute_framework=PyArrowTable.get_class_name(),
            )
        ],
        compute_frameworks=[PyArrowTable],
        output_framework=PandasDataFrame,
        plugin_collector=PLUGINS,
    )
    root = next(step for step in plan if step.feature_group is NanNull2964Root)
    consumer = next(step for step in plan if step.feature_group is NanNull2964Consumer)

    assert root.result_framework is None
    assert not _potential_loss(root)
    assert consumer.result_framework is PandasDataFrame
    assert _potential_loss(consumer) is True


@pytest.mark.parametrize(
    ("native_type", "framework", "expected"),
    [
        pytest.param(pa.Table, PyArrowTable, False, id="native-arrow"),
        pytest.param(pa.Table, PandasDataFrame, True, id="arrow-import-to-pandas"),
        pytest.param(pa.Table, SqliteFramework, True, id="arrow-import-to-sqlite"),
        pytest.param(dict, PythonDictFramework, False, id="native-dict"),
        pytest.param(dict, PandasDataFrame, True, id="dict-import-to-pandas"),
        pytest.param(dict, SqliteFramework, True, id="dict-import-to-sqlite"),
        pytest.param(pd.DataFrame, PandasDataFrame, False, id="native-pandas"),
        pytest.param(pd.DataFrame, SqliteFramework, True, id="pandas-import-to-sqlite"),
        pytest.param(SqliteRelation, SqliteFramework, False, id="native-sqlite"),
        pytest.param(SqliteRelation, PandasDataFrame, False, id="collapsed-import-to-pandas"),
        pytest.param(NanNull2964SqliteSubtype, SqliteFramework, False, id="native-sqlite-subtype"),
        pytest.param(NanNull2964GenericSqlite[int], SqliteFramework, False, id="native-sqlite-generic-origin"),
        pytest.param(dict[str, list[Any]], PythonDictFramework, False, id="native-dict-generic-origin"),
        pytest.param(None, PandasDataFrame, True, id="unknown-import-to-pandas"),
        pytest.param(None, SqliteFramework, True, id="unknown-import-to-sqlite"),
        pytest.param(None, PyArrowTable, False, id="unknown-import-to-arrow"),
    ],
)
def test_neutral_root_import_uses_type_metadata_without_loading(
    native_type: Any, framework: type[ComputeFramework], expected: bool
) -> None:
    plan = _neutral_plan(native_type, framework)

    assert len(plan) == 1
    step = plan[0]
    assert step.step_kind == "compute"
    assert step.implementation_path == "neutral"
    assert step.compute_framework is framework
    assert step.source_compute_framework is None
    assert _potential_loss(step) is expected


@pytest.mark.parametrize(
    ("framework", "expected"),
    [(PythonDictFramework, False), (PyArrowTable, False), (PandasDataFrame, True), (SqliteFramework, True)],
)
def test_api_data_uses_the_existing_neutral_type_hook(
    framework: type[ComputeFramework], expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("explain must not read api_data values")

    monkeypatch.setattr(ApiDataFG, "load_neutral", classmethod(forbid_load))
    plan = mloda.explain(
        [NEUTRAL],
        compute_frameworks=[framework],
        api_data={"nan_null_fixture": {NEUTRAL: [float("nan"), None, 1.0]}},
        plugin_collector=PluginCollector.enabled_feature_groups({ApiDataFG}),
    )

    assert len(plan) == 1
    assert plan[0].implementation_path == "neutral"
    assert _potential_loss(plan[0]) is expected


def test_arrow_backed_pandas_retains_valid_nan_before_the_potential_sqlite_import() -> None:
    values = pa.array([float("nan"), None, 1.0], type=pa.float64(), from_pandas=False)
    frame = pa.table({NEUTRAL: values}).to_pandas(types_mapper=pd.ArrowDtype)
    restored = pa.Table.from_pandas(frame, preserve_index=False)[NEUTRAL].to_pylist()

    assert math.isnan(restored[0])
    assert restored[1] is None
    assert restored[2] == 1.0
    assert str(frame.dtypes[NEUTRAL]) == "double[pyarrow]"
    plan = _neutral_plan(type(frame), SqliteFramework, frame)
    assert _potential_loss(plan[0]) is True


def test_capability_changes_only_affect_new_plan_step_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _transfer_plan(PyArrowTable, PandasDataFrame)
    before = next(step for step in plan if step.step_kind == "transform")
    assert _potential_loss(before) is True

    monkeypatch.setattr(PandasDataFrame, "nan_null_semantics", classmethod(lambda cls: "distinct"))
    rebuilt = _transfer_plan(PyArrowTable, PandasDataFrame)
    after = next(step for step in rebuilt if step.step_kind == "transform")

    assert before.compute_framework is after.compute_framework is PandasDataFrame
    assert _potential_loss(before) is True
    assert _potential_loss(after) is False


@pytest.mark.parametrize("kind", ["compute", "join", "transform"])
def test_false_loss_metadata_preserves_the_version_nine_lock(kind: StepKind) -> None:
    ordinary = _manual_step(kind)
    explicit_false = _manual_step(kind, False)

    assert PLAN_LOCK_FORMAT == 9
    assert json.loads(_lock_text([ordinary]))["format"] == 9
    assert "potential_nan_null_loss" not in _lock_text([ordinary])
    assert _lock_text([ordinary]) == _lock_text([explicit_false])


@pytest.mark.parametrize("kind", ["compute", "join", "transform"])
def test_loss_metadata_changes_structure_hash_but_not_wide_content_hash(kind: StepKind) -> None:
    ordinary = _manual_step(kind)
    flagged = _manual_step(kind, True)
    section = {"compute": "compute", "join": "joins", "transform": "transforms"}[kind]
    flagged_lock, flagged_wide = _build([flagged])
    ordinary_lock, ordinary_wide = _build([ordinary])

    assert flagged_lock[section][0]["potential_nan_null_loss"] is True
    assert "potential_nan_null_loss" not in ordinary_lock[section][0]
    assert flagged_wide == ordinary_wide
    assert plan_structure_hash([flagged]) != plan_structure_hash([ordinary])
    assert plan_content_hash([flagged]) == plan_content_hash([ordinary])


@pytest.mark.parametrize("kind", ["compute", "join", "transform"])
def test_plan_lock_detects_a_changed_loss_advisory(kind: StepKind, tmp_path: Path) -> None:
    ordinary = [_manual_step(kind)]
    flagged = [_manual_step(kind, True)]
    path = tmp_path / "nan-null.plan.lock"
    write_plan_lock(ordinary, path)

    with pytest.raises(PlanLockMismatchError, match="potential_nan_null_loss"):
        check_plan_lock(flagged, path)
    write_plan_lock(flagged, path)
    check_plan_lock(flagged, path)
    with pytest.raises(PlanLockMismatchError, match="potential_nan_null_loss"):
        check_plan_lock(ordinary, path)


@pytest.mark.parametrize("kind", ["compute", "join", "transform"])
@pytest.mark.parametrize("potential_loss", [False, True])
def test_service_step_record_emits_an_explicit_boolean(kind: StepKind, potential_loss: bool) -> None:
    step = _manual_step(kind, True) if potential_loss else _manual_step(kind)
    record = _step_record(1, step, None)

    assert record["potential_nan_null_loss"] is potential_loss


JOIN_KEY = "nan_null_2964_join_key"
JOIN_LEFT = "nan_null_2964_join_left"
JOIN_RIGHT = "nan_null_2964_join_right"
JOIN_CONSUMER = "nan_null_2964_join_consumer"
JOIN_DESTINATION = "join_destination_framework_2964"
JOIN_CASES = [
    pytest.param(PyArrowTable, PandasDataFrame, True, id="distinct-to-dtype-dependent"),
    pytest.param(PyArrowTable, SqliteFramework, True, id="distinct-to-collapsed"),
    pytest.param(SqliteFramework, PandasDataFrame, False, id="collapsed-to-dtype-dependent"),
    pytest.param(PandasDataFrame, PandasDataFrame, False, id="same-pandas-native"),
    pytest.param(SqliteFramework, SqliteFramework, False, id="same-sqlite-native"),
]


class NanNull2964JoinLeft(NanNull2964Root):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {JOIN_KEY, JOIN_LEFT}

    @classmethod
    def index_columns(cls) -> list[Any] | None:
        from mloda.user import Index

        return [Index((JOIN_KEY,))]


class NanNull2964JoinRight(NanNull2964JoinLeft):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {JOIN_KEY, JOIN_RIGHT}


class NanNull2964JoinConsumer(NanNull2964Consumer):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {JOIN_CONSUMER}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {
            Feature(JOIN_LEFT, compute_framework=options.get(JOIN_DESTINATION)),
            Feature(JOIN_RIGHT, compute_framework=options.get(SOURCE_FRAMEWORK)),
        }


def _nan_null_join_link() -> Any:
    from mloda.user import JoinSpec, Link

    return Link.inner(JoinSpec(NanNull2964JoinLeft, JOIN_KEY), JoinSpec(NanNull2964JoinRight, JOIN_KEY))


def _raw_nan_null_join(source: type[ComputeFramework], destination: type[ComputeFramework]) -> Any:
    from mloda.core.core.step.join_step import JoinStep

    return JoinStep(_nan_null_join_link(), destination, source, set(), set(), set())


def _public_nan_null_join(source: type[ComputeFramework], destination: type[ComputeFramework]) -> list[PlanStep]:
    from contextlib import closing
    import sqlite3

    context = {
        SOURCE_FRAMEWORK: source.get_class_name(),
        JOIN_DESTINATION: destination.get_class_name(),
    }
    if destination is SqliteFramework:
        context["connection_handle"] = "nan_null_join"
    request = Feature(
        JOIN_CONSUMER,
        Options(context=context),
        compute_framework=destination.get_class_name(),
    )
    with closing(sqlite3.connect(":memory:")) as connection:
        return mloda.explain(
            [request],
            compute_frameworks=[source, destination],
            links={_nan_null_join_link()},
            data_access_collection=DataAccessCollection(connections={"nan_null_join": connection}),
            plugin_collector=PluginCollector.enabled_feature_groups(
                {NanNull2964JoinLeft, NanNull2964JoinRight, NanNull2964JoinConsumer}
            ),
        )


@pytest.mark.parametrize(("source", "destination", "expected"), JOIN_CASES)
def test_real_join_step_builder_derives_loss_from_its_frameworks(
    source: type[ComputeFramework], destination: type[ComputeFramework], expected: bool
) -> None:
    from mloda.core.api.plan_info import build_plan_steps
    from mloda.core.core.step.join_step import JoinStep

    raw_join = _raw_nan_null_join(source, destination)
    assert isinstance(raw_join, JoinStep)
    plan = build_plan_steps([raw_join])

    assert len(plan) == 1
    join = plan[0]
    assert join.step_kind == "join"
    assert join.compute_framework is destination
    assert join.source_compute_framework is source
    assert join.join_keys == (f"{JOIN_KEY}={JOIN_KEY}",)
    assert _potential_loss(join) is expected


@pytest.mark.parametrize(("source", "destination", "expected"), JOIN_CASES)
def test_public_join_plan_reports_the_join_route_and_its_conversion(
    source: type[ComputeFramework], destination: type[ComputeFramework], expected: bool
) -> None:
    plan = _public_nan_null_join(source, destination)
    joins = [step for step in plan if step.step_kind == "join"]
    transforms = [step for step in plan if step.step_kind == "transform"]

    assert len(joins) == 1
    join = joins[0]
    assert join.compute_framework is destination
    assert join.source_compute_framework is source
    assert join.join_keys == (f"{JOIN_KEY}={JOIN_KEY}",)
    if source is destination:
        assert transforms == []
    else:
        assert len(transforms) == 1
        assert transforms[0].compute_framework is destination
        assert transforms[0].source_compute_framework is source
    assert all(not _potential_loss(step) for step in plan if step.step_kind == "compute")
    assert _potential_loss(join) is expected
    assert all(_potential_loss(step) is expected for step in transforms)


@pytest.mark.parametrize("destination", [PandasDataFrame, SqliteFramework])
def test_real_join_between_different_framework_classes_with_the_same_native_type_is_quiet(
    destination: type[ComputeFramework], monkeypatch: pytest.MonkeyPatch
) -> None:
    from mloda.core.api.plan_info import build_plan_steps

    native = destination.expected_data_framework()
    monkeypatch.setattr(PyArrowTable, "expected_data_framework", classmethod(lambda cls: native))
    assert PyArrowTable is not destination

    plan = build_plan_steps([_raw_nan_null_join(PyArrowTable, destination)])

    assert _potential_loss(plan[0]) is False


class NanNull2964CollapsedPandasAlias(ComputeFramework):
    @classmethod
    def expected_data_framework(cls) -> Any:
        return pd.DataFrame

    @classmethod
    def nan_null_semantics(cls) -> Literal["collapsed"]:
        return "collapsed"

    @staticmethod
    def is_available() -> bool:
        return False


def test_neutral_pandas_type_shared_with_a_collapsed_plugin_remains_potentially_lossy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mloda.core.abstract_plugins.compute_framework import compute_framework_subclasses

    monkeypatch.setattr(NanNull2964CollapsedPandasAlias, "is_available", staticmethod(lambda: True))
    assert NanNull2964CollapsedPandasAlias in compute_framework_subclasses()
    assert NanNull2964CollapsedPandasAlias.expected_data_framework() is PandasDataFrame.expected_data_framework()
    assert NanNull2964CollapsedPandasAlias.nan_null_semantics() == "collapsed"
    array = pa.array([float("nan"), None, 1.0], type=pa.float64(), from_pandas=False)
    frame = pa.table({NEUTRAL: array}).to_pandas(types_mapper=pd.ArrowDtype)
    restored = pa.Table.from_pandas(frame, preserve_index=False)[NEUTRAL].to_pylist()
    assert math.isnan(restored[0]) and restored[1] is None and restored[2] == 1.0

    plan = _neutral_plan(type(frame), SqliteFramework, frame)

    assert plan[0].implementation_path == "neutral"
    assert _potential_loss(plan[0]) is True
