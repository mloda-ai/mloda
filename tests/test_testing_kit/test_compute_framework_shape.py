"""check_compute_framework_shape: a conforming framework passes, each broken one fails only its own check."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any
from uuid import UUID

import pytest

from mloda._testing.compute_framework_shape import (
    applicable_aspects,
    assert_compute_framework_shape,
    check_compute_framework_shape,
    serves_asof_merge,
)
from mloda.core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.filter.filter_engine import BaseFilterEngine

Factory = Callable[[], type[ComputeFramework]]


def make_empty() -> type[ComputeFramework]:
    class CfShapeEmpty(ComputeFramework):
        pass

    return CfShapeEmpty


def make_unavailable() -> type[ComputeFramework]:
    class CfShapeUnavailable(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

        @classmethod
        def filter_engine(cls) -> Any:
            raise ImportError("optional dependency missing")

    return CfShapeUnavailable


def make_init_needs_argument() -> type[ComputeFramework]:
    class CfShapeNeedsArg(ComputeFramework):
        def __init__(self, required: int) -> None:
            super().__init__()

    return CfShapeNeedsArg


def make_init_ignores_uuid() -> type[ComputeFramework]:
    class CfShapeIgnoresUuid(ComputeFramework):
        def __init__(self, uuid: UUID | None = None) -> None:
            super().__init__()

    return CfShapeIgnoresUuid


def make_is_available_not_bool() -> type[ComputeFramework]:
    class CfShapeAvailStr(ComputeFramework):
        @staticmethod
        def is_available() -> Any:
            return "yes"

    return CfShapeAvailStr


def make_expected_data_framework_bad() -> type[ComputeFramework]:
    class CfShapeBadData(ComputeFramework):
        @classmethod
        def expected_data_framework(cls) -> Any:
            return 3

    return CfShapeBadData


def make_filter_engine_not_a_class() -> type[ComputeFramework]:
    class CfShapeBadFilter(ComputeFramework):
        @classmethod
        def filter_engine(cls) -> Any:
            return int

    return CfShapeBadFilter


def make_merge_engine_raises() -> type[ComputeFramework]:
    class CfShapeBadMerge(ComputeFramework):
        @classmethod
        def merge_engine(cls) -> Any:
            raise ValueError("boom")

    return CfShapeBadMerge


def make_mask_engine_bad() -> type[ComputeFramework]:
    class CfShapeBadMask(ComputeFramework):
        @classmethod
        def mask_engine(cls) -> Any:
            return object()

    return CfShapeBadMask


def make_uuid_not_a_uuid() -> type[ComputeFramework]:
    class CfShapeStrUuid(ComputeFramework):
        def __init__(self, uuid: UUID | None = None) -> None:
            super().__init__()
            self.uuid: Any = "not-a-uuid"

    return CfShapeStrUuid


def make_constant_uuid() -> type[ComputeFramework]:
    pinned = UUID("12345678-1234-5678-1234-567812345678")

    class CfShapeConstUuid(ComputeFramework):
        def __init__(self, uuid: UUID | None = None) -> None:
            super().__init__()
            self.uuid = pinned if uuid is None else uuid

    return CfShapeConstUuid


def make_mask_engine_not_implemented() -> type[ComputeFramework]:
    class CfShapeMaskNotImpl(ComputeFramework):
        @classmethod
        def mask_engine(cls) -> Any:
            raise NotImplementedError("no mask")

    return CfShapeMaskNotImpl


def make_filter_engine_instance_method() -> type[ComputeFramework]:
    class CfShapeInstanceFilter(ComputeFramework):
        def filter_engine(self) -> Any:  # type: ignore[override]
            return BaseFilterEngine

    return CfShapeInstanceFilter


def test_empty_framework_passes_with_no_aspects() -> None:
    cls = make_empty()
    result = check_compute_framework_shape(cls)
    assert result.check == "compute_framework_shape"
    assert result.subject == "CfShapeEmpty"
    assert result.failures == ()
    assert applicable_aspects(cls) == ()


def test_unavailable_framework_passes_and_is_not_probed() -> None:
    cls = make_unavailable()
    assert check_compute_framework_shape(cls).failures == ()
    assert applicable_aspects(cls) == ()


# (module, class name, dependency to importorskip, expected aspects)
BUILT_INS = [
    ("mloda.user.python_dict", "PythonDictFramework", None, ("filter", "mask", "merge")),
    ("mloda.user.pyarrow", "PyArrowTable", "pyarrow", ("filter", "mask", "merge")),
    ("mloda.user.pandas", "PandasDataFrame", "pandas", ("filter", "mask", "merge")),
    ("mloda.user.polars", "PolarsDataFrame", "polars", ("filter", "mask", "merge")),
    ("mloda.user.polars", "PolarsLazyDataFrame", "polars", ("filter", "mask", "merge")),
    ("mloda.user.duckdb", "DuckDBFramework", "duckdb", ("filter", "mask", "merge")),
    ("mloda.user.sqlite", "SqliteFramework", "pyarrow", ("filter", "mask", "merge")),
    ("mloda.user.iceberg", "IcebergFramework", "pyiceberg", ("filter",)),
]


@pytest.mark.parametrize(("module", "name", "dependency", "aspects"), BUILT_INS, ids=[b[1] for b in BUILT_INS])
def test_built_in_framework_passes(module: str, name: str, dependency: str | None, aspects: tuple[str, ...]) -> None:
    if dependency is not None:
        pytest.importorskip(dependency)
    cls = getattr(importlib.import_module(module), name)
    if not cls.is_available():
        pytest.skip(f"{name} is not available")
    assert check_compute_framework_shape(cls).failures == ()
    assert applicable_aspects(cls) == aspects


@pytest.mark.parametrize(
    ("make", "prefix", "message"),
    [
        (make_init_needs_argument, "construct:", "required"),
        (make_init_ignores_uuid, "construct:", "uuid"),
        (make_is_available_not_bool, "is_available:", "bool"),
        (make_expected_data_framework_bad, "expected_data_framework:", "type"),
        (make_filter_engine_not_a_class, "filter_engine:", "BaseFilterEngine"),
        (make_merge_engine_raises, "merge_engine:", "boom"),
        (make_mask_engine_bad, "mask_engine:", "BaseMaskEngine"),
        (make_uuid_not_a_uuid, "construct:", "UUID"),
        (make_constant_uuid, "construct:", "share"),
        (make_mask_engine_not_implemented, "mask_engine:", "None or"),
        (make_filter_engine_instance_method, "filter_engine:", "classmethod"),
    ],
)
def test_broken_framework_fails_only_its_check(make: Factory, prefix: str, message: str) -> None:
    result = check_compute_framework_shape(make())
    assert len(result.failures) == 1, result.failures
    assert result.failures[0].startswith(prefix), result.failures
    assert message in result.failures[0], result.failures


def test_assert_passes_silently_for_conforming_framework() -> None:
    assert_compute_framework_shape(make_empty())


def test_assert_raises_pytest_failure_for_broken_framework() -> None:
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_compute_framework_shape(make_init_ignores_uuid())
    assert "construct:" in str(excinfo.value)


def _framework_with_merge(engine: type[BaseMergeEngine] | None, available: bool = True) -> type[ComputeFramework]:
    class CfAsofStub(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return available

        @classmethod
        def merge_engine(cls) -> Any:
            if engine is None:
                raise NotImplementedError("no merge")
            return engine

    return CfAsofStub


class _EngineWithAsof(BaseMergeEngine):
    def merge_asof(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _EngineWithoutAsof(BaseMergeEngine):
    pass


def test_serves_asof_merge_requires_an_overridden_merge_asof() -> None:
    assert serves_asof_merge(_framework_with_merge(_EngineWithAsof)) is True
    assert serves_asof_merge(_framework_with_merge(_EngineWithoutAsof)) is False


def test_serves_asof_merge_false_without_merge_or_when_unavailable() -> None:
    assert serves_asof_merge(_framework_with_merge(None)) is False
    assert serves_asof_merge(_framework_with_merge(_EngineWithAsof, available=False)) is False
