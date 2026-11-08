"""Typed host for ComputeFramework conformance suites: adapter lifecycle, frame building and result reading."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from mloda._testing.compute_framework_shape import applicable_aspects
from mloda._testing.declarations import Cause
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.suite import ConformanceSuite, HookError, SwitchedOffError
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.user import ParallelizationMode

if TYPE_CHECKING:
    import pyarrow as pa


class ComputeFrameworkSuite(ConformanceSuite, suite_base=True):
    """Host sets adapter_class; a suite base sets aspect and may list required_hooks a host must override."""

    adapter_class: ClassVar[type[FrameworkAdapter]]
    aspect: ClassVar[str]
    required_hooks: ClassVar[tuple[str, ...]] = ()
    case_ids: ClassVar[frozenset[str]] = frozenset()
    unsupported_cases: ClassVar[Mapping[str, tuple[Cause, str]]] = MappingProxyType({})

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
            invalid = _invalid_unsupported_cases(cls)
            if invalid:
                raise SwitchedOffError(f"{cls.__name__}: " + "; ".join(invalid))
        super().__init_subclass__(suite_base=suite_base, **kwargs)

    @pytest.fixture
    def adapter(self) -> Iterator[FrameworkAdapter]:
        instance = self.adapter_class()
        if not instance.is_available():
            pytest.skip(f"{instance.compute_framework().__name__} is not available")
        with instance:
            yield instance

    @pytest.fixture
    def framework_instance(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> ComputeFramework:
        instance = framework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        if adapter.connection is not None:
            instance.set_framework_connection_object(adapter.connection)
        return instance

    @pytest.fixture
    def framework(self, adapter: FrameworkAdapter) -> type[ComputeFramework]:
        return self.served_framework(adapter)

    def served_framework(self, adapter: FrameworkAdapter) -> type[ComputeFramework]:
        """The adapter's framework; fails the test when the host's framework does not serve the aspect."""
        framework = adapter.compute_framework()
        if not self._serves(adapter, framework):
            pytest.fail(f"{framework.__name__} does not serve the '{self.aspect}' aspect; this host is mis-wired")
        return framework

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return self.aspect in applicable_aspects(framework)

    def arrow_types(self, name: str) -> Mapping[str, pa.DataType] | None:
        return None

    def frame(self, adapter: FrameworkAdapter, name: str, data: TypedTestData) -> Any:
        if adapter.capabilities.typed_columns:
            return adapter.from_typed(data, self.arrow_types(name))
        return adapter.from_typed(data)

    def duplicate_column_frame(self, adapter: FrameworkAdapter, first_type: DataType, second_type: DataType) -> Any:
        """A frame whose two columns are both named "a", typed first_type then second_type."""
        return adapter.duplicate_column_frame(first_type, second_type)

    def skip_unsupported_case(self, case_id: str) -> None:
        """Skip the case with the host's declared reason when the backend cannot hold it."""
        entry = self.unsupported_cases.get(case_id)
        if entry is not None:
            pytest.skip(f"{entry[0].name}: {entry[1]}")

    def column_values(self, adapter: FrameworkAdapter, result: Any, column: str) -> list[Any]:
        return adapter.column_values(result, column)

    def rows(self, adapter: FrameworkAdapter, result: Any) -> list[dict[str, Any]]:
        return adapter.rows(result)

    def row_count(self, adapter: FrameworkAdapter, result: Any) -> int:
        return adapter.row_count(result)

    def _result_rows(self, adapter: FrameworkAdapter, frame: Any) -> list[dict[str, Any]]:
        """run_all hands relation and lazy results back as Arrow tables or eager frames, which adapters cannot read."""
        if hasattr(frame, "to_pylist"):
            return list(frame.to_pylist())
        if hasattr(frame, "to_dicts"):
            return list(frame.to_dicts())
        return self.rows(adapter, frame)

    def _result_row_count(self, adapter: FrameworkAdapter, frame: Any) -> int:
        if hasattr(frame, "num_rows"):
            return int(frame.num_rows)
        if hasattr(frame, "height"):
            return int(frame.height)
        return self.row_count(adapter, frame)


def _invalid_unsupported_cases(cls: type[ComputeFrameworkSuite]) -> list[str]:
    cases = cls.unsupported_cases
    if not isinstance(cases, Mapping):
        return [f"unsupported_cases must be a mapping of case id to (Cause, reason), got {cases!r}"]
    problems = []
    for case_id, entry in cases.items():
        if case_id not in cls.case_ids:
            problems.append(f"unsupported_cases key '{case_id}' is not a case of this suite")
        elif not (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], Cause)
            and isinstance(entry[1], str)
            and entry[1]
        ):
            problems.append(f"unsupported_cases['{case_id}'] must be (Cause, reason), got {entry!r}")
    return problems


def _is_unset(cls: type, name: str) -> bool:
    for base in cls.__mro__:
        if name in base.__dict__:
            return bool(base.__dict__.get("_suite_base", False))
    return True
