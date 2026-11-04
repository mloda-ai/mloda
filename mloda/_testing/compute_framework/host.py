"""Typed host for ComputeFramework conformance suites: adapter lifecycle, frame building and result reading."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from mloda._testing.compute_framework_shape import applicable_aspects
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.suite import ConformanceSuite, HookError
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.compute_framework import ComputeFramework

if TYPE_CHECKING:
    import pyarrow as pa


class ComputeFrameworkSuite(ConformanceSuite, suite_base=True):
    """Host sets adapter_class; a suite base sets aspect and may list required_hooks a host must override."""

    adapter_class: ClassVar[type[FrameworkAdapter]]
    aspect: ClassVar[str]
    required_hooks: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, suite_base: bool = False, **kwargs: Any) -> None:
        if not suite_base:
            problems = []
            if getattr(cls, "adapter_class", None) is None:
                problems.append("a host must set adapter_class")
            missing = [name for name in cls.required_hooks if _is_unset(cls, name)]
            if missing:
                problems.append(f"a host must override: {', '.join(missing)}")
            if problems:
                raise HookError(f"{cls.__name__}: " + "; ".join(problems))
        super().__init_subclass__(suite_base=suite_base, **kwargs)

    @pytest.fixture
    def adapter(self) -> Iterator[FrameworkAdapter]:
        instance = self.adapter_class()
        if not instance.is_available():
            pytest.skip(f"{instance.compute_framework().__name__} is not available")
        with instance:
            yield instance

    @pytest.fixture
    def framework(self, adapter: FrameworkAdapter) -> type[ComputeFramework]:
        framework = adapter.compute_framework()
        if self.aspect not in applicable_aspects(framework):
            pytest.fail(f"{framework.__name__} does not serve the '{self.aspect}' aspect; this host is mis-wired")
        return framework

    def arrow_types(self, name: str) -> Mapping[str, pa.DataType] | None:
        return None

    def frame(self, adapter: FrameworkAdapter, name: str, data: TypedTestData) -> Any:
        return adapter.from_typed(data, self.arrow_types(name))

    def values(self, adapter: FrameworkAdapter, result: Any, column: str) -> list[Any]:
        return adapter.column_values(result, column)

    def row_count(self, adapter: FrameworkAdapter, result: Any) -> int:
        return adapter.row_count(result)


def _is_unset(cls: type, name: str) -> bool:
    for base in cls.__mro__:
        if name in base.__dict__:
            return bool(base.__dict__.get("_suite_base", False))
    return True
