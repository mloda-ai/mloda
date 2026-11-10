"""Loads-into-framework suites: a file, database or document source group loads into a host's compute framework."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.source.source_suite import LoadCapture, expected_implementation_version
from mloda._testing.suite import ConformanceSuite, HookError
from mloda.user import DataAccessCollection, Feature, GlobalFilter, Options, PluginCollector, mloda

if TYPE_CHECKING:
    from mloda.provider import ReadDBFG, ReadDocumentFG, ReadFileFG


class LoadsIntoFrameworkTestMixin(ConformanceSuite, suite_base=True):
    """Host sets adapter_class; tests run the group against that adapter's compute framework."""

    adapter_class: ClassVar[type[FrameworkAdapter]]

    def __init_subclass__(cls, suite_base: bool = False, **kwargs: Any) -> None:
        if not suite_base and getattr(cls, "adapter_class", None) is None:
            raise HookError(f"{cls.__name__}: a host must set adapter_class")
        super().__init_subclass__(suite_base=suite_base, **kwargs)

    @pytest.fixture
    def adapter(self) -> Iterator[FrameworkAdapter]:
        instance = self.adapter_class()
        if not instance.is_available():
            pytest.skip(f"{instance.compute_framework().__name__} is not available")
        with instance:
            yield instance

    def columns_of(self, adapter: FrameworkAdapter, result: Any) -> set[str]:
        return set(adapter.column_names(result))

    def values_of(self, adapter: FrameworkAdapter, result: Any, column: str) -> list[Any]:
        return adapter.column_values(result, column)


class FileLoadsIntoFrameworkTestMixin(LoadsIntoFrameworkTestMixin, suite_base=True):
    """A file group loads exactly the requested columns into the adapter's framework."""

    feature_group_class: type[ReadFileFG]
    columns: ClassVar[tuple[str, ...]] = ("fwloads_a", "fwloads_b", "fwloads_c")

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        raise NotImplementedError

    def test_loads_requested_columns(self, adapter: FrameworkAdapter, tmp_path: Path) -> None:
        path = tmp_path / f"fwloads{self.feature_group_class.suffixes()[0]}"
        self.write_file(path, {name: [1, 2, 3] for name in self.columns})

        result = mloda.run_all(
            list(self.columns[:2]),
            compute_frameworks=[adapter.compute_framework()],
            plugin_collector=PluginCollector.enabled_feature_groups({self.feature_group_class}),
            data_access_collection=DataAccessCollection(files={"fwloads_handle": os.fspath(path)}),
        )

        assert len(result) == 1
        assert self.columns_of(adapter, result[0]) == set(self.columns[:2])


class FileLoaderNameTestMixin(FileLoadsIntoFrameworkTestMixin, suite_base=True):
    """Adds: the INPUT_DATA_LOAD hook names the loader the group used for the adapter's framework."""

    expected_loader: str

    def test_input_data_load_names_the_loader(self, adapter: FrameworkAdapter, tmp_path: Path) -> None:
        path = tmp_path / f"loadername{self.feature_group_class.suffixes()[0]}"
        self.write_file(path, {name: [1, 2, 3] for name in self.columns})
        capture = LoadCapture()

        mloda.run_all(
            list(self.columns[:2]),
            compute_frameworks=[adapter.compute_framework()],
            plugin_collector=PluginCollector.enabled_feature_groups({self.feature_group_class}),
            data_access_collection=DataAccessCollection(files={"loadername_handle": os.fspath(path)}),
            function_extender={capture},
        )

        assert len(capture.contexts) == 1
        context = capture.contexts[0]
        assert context.reader_class is self.feature_group_class
        assert context.data_access_loader == self.expected_loader
        assert context.implementation == self.expected_loader
        assert context.implementation_version == expected_implementation_version(self.expected_loader)


class DatabaseLoadsIntoFrameworkTestMixin(LoadsIntoFrameworkTestMixin, suite_base=True):
    """A database group loads exactly the requested columns, with their values, into the adapter's framework."""

    feature_group_class: type[ReadDBFG]
    credential_key: str
    expected_loader: str
    columns: ClassVar[tuple[str, ...]] = ("dbloads_a", "dbloads_b", "dbloads_c")

    def write_database(self, path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
        raise NotImplementedError

    def _run(self, adapter: FrameworkAdapter, tmp_path: Path, extender: Any = None) -> list[Any]:
        path = tmp_path / "dbloads.db"
        self.write_database(path, {"dbloads_table": {name: [1, 2, 3] for name in self.columns}})
        return mloda.run_all(
            list(self.columns[:2]),
            compute_frameworks=[adapter.compute_framework()],
            plugin_collector=PluginCollector.enabled_feature_groups({self.feature_group_class}),
            data_access_collection=DataAccessCollection(
                credentials={"dbloads_handle": {self.credential_key: str(path)}}
            ),
            function_extender=None if extender is None else {extender},
        )

    def test_loads_requested_columns_with_their_values(self, adapter: FrameworkAdapter, tmp_path: Path) -> None:
        result = self._run(adapter, tmp_path)

        assert len(result) == 1
        assert self.columns_of(adapter, result[0]) == set(self.columns[:2])
        assert [self.values_of(adapter, result[0], name) for name in self.columns[:2]] == [[1, 2, 3], [1, 2, 3]]

    def test_a_global_filter_returns_the_matching_rows(self, adapter: FrameworkAdapter, tmp_path: Path) -> None:
        path = tmp_path / "dbloads_filtered.db"
        self.write_database(path, {"dbloads_table": {name: [1, 2, 3, 4, 5, 6] for name in self.columns}})
        global_filter = GlobalFilter()
        global_filter.add_filter("dbloads_a", "range", {"min": 2, "max": 4, "max_exclusive": True})

        result = mloda.run_all(
            list(self.columns[:2]),
            compute_frameworks=[adapter.compute_framework()],
            plugin_collector=PluginCollector.enabled_feature_groups({self.feature_group_class}),
            data_access_collection=DataAccessCollection(
                credentials={"dbloads_handle": {self.credential_key: str(path)}}
            ),
            global_filter=global_filter,
        )

        assert [self.values_of(adapter, result[0], name) for name in self.columns[:2]] == [[2, 3], [2, 3]]

    def test_input_data_load_names_the_loader(self, adapter: FrameworkAdapter, tmp_path: Path) -> None:
        capture = LoadCapture()

        self._run(adapter, tmp_path, capture)

        assert len(capture.contexts) == 1
        context = capture.contexts[0]
        assert context.reader_class is self.feature_group_class
        assert context.data_access_loader == self.expected_loader
        assert context.implementation == self.expected_loader
        assert context.implementation_version == expected_implementation_version(self.expected_loader)


class DocumentLoadsIntoFrameworkTestMixin(LoadsIntoFrameworkTestMixin, suite_base=True):
    """A document group loads exactly the requested ``~`` outputs, with their values, into the adapter's framework."""

    feature_group_class: type[ReadDocumentFG]
    suffix: str
    context: ClassVar[Mapping[str, Any]] = MappingProxyType({})
    text = "framework loads\nbody\n"
    expected_text: ClassVar[str | None] = None

    def _run(self, adapter: FrameworkAdapter, tmp_path: Path, names: list[str]) -> tuple[list[Any], Path]:
        path = tmp_path / f"fwdoc{self.suffix}"
        path.write_text(self.text, encoding="utf-8")
        group_name = self.feature_group_class.get_class_name()
        features: list[Feature | str] = [
            Feature(f"{group_name}{name}", Options(context=dict(self.context))) for name in names
        ]
        result = mloda.run_all(
            features,
            compute_frameworks=[adapter.compute_framework()],
            plugin_collector=PluginCollector.enabled_feature_groups({self.feature_group_class}),
            data_access_collection=DataAccessCollection(files={"fwdoc_handle": os.fspath(path)}),
        )
        return result, path

    def test_loads_all_three_outputs_with_their_values(self, adapter: FrameworkAdapter, tmp_path: Path) -> None:
        name = self.feature_group_class.get_class_name()
        result, path = self._run(adapter, tmp_path, ["", "~source", "~file_type"])

        assert len(result) == 1
        assert self.columns_of(adapter, result[0]) == {name, f"{name}~source", f"{name}~file_type"}
        assert self.values_of(adapter, result[0], f"{name}~source") == [os.fspath(path)]
        assert self.values_of(adapter, result[0], f"{name}~file_type") == [self.suffix.lstrip(".").lower()]
        expected = self.text if self.expected_text is None else self.expected_text
        assert self.values_of(adapter, result[0], name) == [expected]

    def test_loads_only_the_requested_outputs(self, adapter: FrameworkAdapter, tmp_path: Path) -> None:
        name = self.feature_group_class.get_class_name()
        result, path = self._run(adapter, tmp_path, ["~source"])

        assert len(result) == 1
        assert self.columns_of(adapter, result[0]) == {f"{name}~source"}
        assert self.values_of(adapter, result[0], f"{name}~source") == [os.fspath(path)]
