"""Loads-into-framework hosts for SparkFramework: a file, database and document group load into Spark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mloda._testing.source.loads_into_framework_suite import (
    DatabaseLoadsIntoFrameworkTestMixin,
    DocumentLoadsIntoFrameworkTestMixin,
    FileLoadsIntoFrameworkTestMixin,
)
from tests.mixins.reader_feature_groups.format_file_writers import write_csv
from tests.mixins.reader_feature_groups.sqlite_loads_helper import SqliteLoadsHelper
from tests.mixins.reader_feature_groups.lazy_source_group import lazy_document_group, lazy_group
from tests.test_plugins.compute_framework.base_implementations.spark.spark_session_adapter import SparkSessionAdapter


class TestCsvIntoSpark(SparkSessionAdapter, FileLoadsIntoFrameworkTestMixin):
    feature_group_class = lazy_group("csv_fg", "CsvFG")

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_csv(path, columns)


class TestSqliteLoadsIntoSpark(SparkSessionAdapter, SqliteLoadsHelper, DatabaseLoadsIntoFrameworkTestMixin):
    expected_loader = "neutral"


class TestTextLoadsIntoSpark(SparkSessionAdapter, DocumentLoadsIntoFrameworkTestMixin):
    feature_group_class = lazy_document_group("text_fg", "TextFG")
    suffix = ".txt"
