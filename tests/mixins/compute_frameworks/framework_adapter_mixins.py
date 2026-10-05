"""Compute-framework adapters and a loads-into-framework check for file format groups.

Adapters resolve the framework class lazily (after importorskip), never at import, class or parametrize time.
Combine one adapter with FileLoadsIntoFrameworkMixin in a concrete Test class.
"""

import os
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.user import DataAccessCollection, PluginCollector, mloda


class FrameworkAdapterMixin:
    """Names one compute framework: its module, its class, and how to read a result's column names."""

    module_name: str

    def required_module(self) -> Any:
        return pytest.importorskip(self.module_name)

    def compute_framework(self) -> type[ComputeFramework]:
        raise NotImplementedError

    def columns_of(self, result: Any) -> set[str]:
        return self.compute_framework().extract_column_names(result)


class PyArrowTableAdapter(FrameworkAdapterMixin):
    module_name = "pyarrow"

    def compute_framework(self) -> type[ComputeFramework]:
        self.required_module()
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

        return PyArrowTable


class PandasDataFrameAdapter(FrameworkAdapterMixin):
    module_name = "pandas"

    def compute_framework(self) -> type[ComputeFramework]:
        self.required_module()
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        return PandasDataFrame


class PolarsDataFrameAdapter(FrameworkAdapterMixin):
    module_name = "polars"

    def compute_framework(self) -> type[ComputeFramework]:
        self.required_module()
        from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

        return PolarsDataFrame


class PythonDictAdapter(FrameworkAdapterMixin):
    module_name = "builtins"

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
            PythonDictFramework,
        )

        return PythonDictFramework


class FileLoadsIntoFrameworkMixin(FrameworkAdapterMixin):
    """A file group loads exactly the requested columns into the adapter's framework.

    The concrete class sets ``file_group`` and ``write_file``; combine with an adapter.
    """

    file_group: type[Any]
    columns = ("fwloads_a", "fwloads_b", "fwloads_c")

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        raise NotImplementedError

    def test_loads_requested_columns(self, tmp_path: Path) -> None:
        framework = self.compute_framework()
        path = tmp_path / f"fwloads{self.file_group.suffixes()[0]}"
        self.write_file(path, {name: [1, 2, 3] for name in self.columns})

        result = mloda.run_all(
            list(self.columns[:2]),
            compute_frameworks=[framework],
            plugin_collector=PluginCollector.enabled_feature_groups({self.file_group}),
            data_access_collection=DataAccessCollection(files={"fwloads_handle": os.fspath(path)}),
        )

        assert len(result) == 1
        assert self.columns_of(result[0]) == set(self.columns[:2])


class _LoadCapture(Extender):
    def __init__(self) -> None:
        self.priority = 100
        self.contexts: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        if context is not None:
            self.contexts.append(context)
        return result


class FileLoaderNameMixin(FileLoadsIntoFrameworkMixin):
    """Adds: the INPUT_DATA_LOAD hook names the loader the group used for the adapter's framework."""

    expected_loader: str

    def test_input_data_load_names_the_loader(self, tmp_path: Path) -> None:
        framework = self.compute_framework()
        path = tmp_path / f"loadername{self.file_group.suffixes()[0]}"
        self.write_file(path, {name: [1, 2, 3] for name in self.columns})
        capture = _LoadCapture()

        mloda.run_all(
            list(self.columns[:2]),
            compute_frameworks=[framework],
            plugin_collector=PluginCollector.enabled_feature_groups({self.file_group}),
            data_access_collection=DataAccessCollection(files={"loadername_handle": os.fspath(path)}),
            function_extender={capture},
        )

        assert len(capture.contexts) == 1
        context = capture.contexts[0]
        assert context.reader_class is self.file_group
        assert context.data_access_loader == self.expected_loader
