"""Pins the per-reader ``READER_OPTIONS`` declarations and makes the declared default load-bearing.

Reader option keys are read at MATCH time (inside ``match_subclass_data_access``), before the
framework materializes any ``PROPERTY_MAPPING`` default, so the readers declare them themselves
through ``ReaderOptionSpec``. See the core-side contract in
``tests/test_core/test_abstract_plugins/test_components/test_reader_option_declarations.py``.

What is pinned here:

* ``ReadFile`` and ``ReadDocument`` declare ``document_suffixes`` (``runtime_default=frozenset()``)
  and ``data_access_handle`` (``runtime_default=None``); ``ReadDB`` declares
  ``data_access_handle``. Shipped concrete readers (``CsvReader``, ``MarkdownDocumentReader``,
  ``SQLITEReader``) inherit the declarations and re-declare nothing.
* Every options key these three readers actually read is a declared key. The reads are observed
  behaviorally through a recording ``Options`` subclass, so the inventory cannot drift.
* The declared ``runtime_default`` is load-bearing, not documentation: the ``document_suffixes``
  fallback in ``ReadFile.match_subclass_data_access`` / ``ReadDocument.match_subclass_data_access``
  comes from the declaration instead of a hard-coded ``frozenset()``. A reader that declares
  ``runtime_default=frozenset({".json"})`` therefore matches ``.json`` differently from a stock
  reader even when the option is not set: ReadFile DECLINES the file (``document_suffixes``
  auto-excludes those suffixes for structured readers) while ReadDocument CLAIMS it.

Isolation: every reader defined here deliberately does NOT override ``load_data``, so
``is_final_reader()`` is False and ``get_all_filtered_subclasses`` never collects it. Matching is
exercised by calling ``match_subclass_data_access`` directly, never through ``mlodaAPI``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.input_data.reader_option_spec import ReaderOptionSpec
from mloda.user import DataAccessCollection, Options
from mloda_plugins.feature_group.input_data.read_db import ReadDB
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from mloda_plugins.feature_group.input_data.read_files.markdown_document_reader import MarkdownDocumentReader


_RESERVED_KEY = "BaseInputData"


class _RodRecordingOptions(Options):
    """Empty Options that records every key read through ``get``."""

    def __init__(self) -> None:
        super().__init__()
        self.read_keys: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        self.read_keys.append(key)
        return super().get(key, default)


class _RodFileProbe(ReadFile):
    """ReadFile probe with an inert suffix; only exists because ``ReadFile.suffix()`` raises."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".rod_unused",)


class _RodStockJsonReadFile(ReadFile):
    """Stock ReadFile reader for ``.json``: inherits the ``frozenset()`` document_suffixes default."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".json",)


class _RodJsonExcludingReadFile(ReadFile):
    """ReadFile reader declaring ``.json`` as a document suffix, so it must decline ``.json`` files."""

    READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
        "document_suffixes": ReaderOptionSpec(
            "Suffixes handed to document readers; declared non-empty so ReadFile auto-excludes them.",
            runtime_default=frozenset({".json"}),
        ),
    }

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".json",)


class _RodStockJsonReadDocument(ReadDocument):
    """Stock ReadDocument reader for ``.json``: skips the structured suffix by default."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".json",)


class _RodJsonClaimingReadDocument(ReadDocument):
    """ReadDocument reader declaring ``.json`` as a document suffix, so it must claim ``.json`` files."""

    READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
        "document_suffixes": ReaderOptionSpec(
            "Structured suffixes this document reader owns; declared non-empty to claim .json.",
            runtime_default=frozenset({".json"}),
        ),
    }

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".json",)


@pytest.fixture
def json_path(tmp_path: Path) -> str:
    """A real ``.json`` file path in an isolated tmp dir."""
    path = tmp_path / "rod_payload.json"
    path.write_text('{"value": 1}', encoding="utf-8")
    return str(path)


@pytest.fixture
def csv_path(tmp_path: Path) -> str:
    """A real ``.csv`` file path in an isolated tmp dir."""
    path = tmp_path / "rod_rows.csv"
    path.write_text("id,amount\n1,10\n", encoding="utf-8")
    return str(path)


class TestReadFileDeclarations:
    """ReadFile declares the two keys its matcher reads, and its concrete readers inherit them."""

    def test_declares_exactly_its_match_time_keys(self) -> None:
        assert ReadFile.declared_reader_option_keys() == {"document_suffixes", "data_access_handle", _RESERVED_KEY}

    def test_declared_runtime_defaults(self) -> None:
        assert ReadFile.reader_option_default("document_suffixes") == frozenset()
        assert ReadFile.reader_option_default("data_access_handle") is None

    def test_csv_reader_inherits_without_redeclaring(self) -> None:
        assert "READER_OPTIONS" not in CsvReader.__dict__
        assert CsvReader.declared_reader_option_keys() == ReadFile.declared_reader_option_keys()
        assert CsvReader.reader_option_default("document_suffixes") == frozenset()
        assert CsvReader.reader_option_default("data_access_handle") is None


class TestReadDocumentDeclarations:
    """ReadDocument declares the same two keys, and its concrete readers inherit them."""

    def test_declares_exactly_its_match_time_keys(self) -> None:
        assert ReadDocument.declared_reader_option_keys() == {"document_suffixes", "data_access_handle", _RESERVED_KEY}

    def test_declared_runtime_defaults(self) -> None:
        assert ReadDocument.reader_option_default("document_suffixes") == frozenset()
        assert ReadDocument.reader_option_default("data_access_handle") is None

    def test_markdown_reader_inherits_without_redeclaring(self) -> None:
        assert "READER_OPTIONS" not in MarkdownDocumentReader.__dict__
        assert MarkdownDocumentReader.declared_reader_option_keys() == ReadDocument.declared_reader_option_keys()
        assert MarkdownDocumentReader.reader_option_default("document_suffixes") == frozenset()


class TestReadDBDeclarations:
    """ReadDB reads only the handle hint, so it declares only that key."""

    def test_declares_exactly_its_match_time_keys(self) -> None:
        assert ReadDB.declared_reader_option_keys() == {"data_access_handle", _RESERVED_KEY}

    def test_declared_runtime_default(self) -> None:
        assert ReadDB.reader_option_default("data_access_handle") is None

    def test_document_suffixes_is_not_a_read_db_key(self) -> None:
        with pytest.raises(ValueError, match="document_suffixes"):
            ReadDB.reader_option_default("document_suffixes")

    def test_sqlite_reader_inherits_without_redeclaring(self) -> None:
        assert "READER_OPTIONS" not in SQLITEReader.__dict__
        assert SQLITEReader.declared_reader_option_keys() == ReadDB.declared_reader_option_keys()
        assert SQLITEReader.reader_option_default("data_access_handle") is None


class TestEveryOptionKeyReadIsDeclared:
    """Observed match-time reads are a subset of the declared keys, per reader family."""

    def test_read_file_reads_only_declared_keys(self, csv_path: str) -> None:
        options = _RodRecordingOptions()
        data_access = DataAccessCollection(files={"rod_rows": csv_path})

        assert _RodFileProbe.match_subclass_data_access(data_access, ["id"], options) is None
        assert set(options.read_keys) == {"document_suffixes", "data_access_handle"}
        assert set(options.read_keys) <= ReadFile.declared_reader_option_keys()

    def test_read_document_reads_only_declared_keys(self, csv_path: str) -> None:
        options = _RodRecordingOptions()
        data_access = DataAccessCollection(files={"rod_rows": csv_path})

        assert ReadDocument.match_subclass_data_access(data_access, ["content"], options) is None
        assert set(options.read_keys) == {"document_suffixes", "data_access_handle"}
        assert set(options.read_keys) <= ReadDocument.declared_reader_option_keys()

    def test_read_db_reads_only_declared_keys(self, tmp_path: Path) -> None:
        options = _RodRecordingOptions()
        data_access = DataAccessCollection(credentials=[{"db_path": str(tmp_path / "rod.sqlite")}])

        assert ReadDB.match_subclass_data_access(data_access, ["any"], options) is None
        assert set(options.read_keys) == {"data_access_handle"}
        assert set(options.read_keys) <= ReadDB.declared_reader_option_keys()


class TestDeclaredDefaultIsLoadBearing:
    """The document_suffixes fallback comes from the declaration, not a hard-coded frozenset()."""

    def test_stock_read_file_claims_a_json_path(self, json_path: str) -> None:
        """Control: the stock ``frozenset()`` default excludes nothing, so ReadFile claims the file."""
        assert _RodStockJsonReadFile.match_subclass_data_access(json_path, ["value"], Options()) == json_path

    def test_declared_default_makes_read_file_decline_json(self, json_path: str) -> None:
        """The declared ``frozenset({".json"})`` default auto-excludes ``.json`` with no option set."""
        assert _RodJsonExcludingReadFile.match_subclass_data_access(json_path, ["value"], Options()) is None

    def test_explicit_option_still_overrides_the_read_file_default(self, json_path: str) -> None:
        """A user-set ``document_suffixes`` wins over the declared default."""
        options = Options({"document_suffixes": frozenset({".json"})})

        assert _RodStockJsonReadFile.match_subclass_data_access(json_path, ["value"], options) is None

    def test_stock_read_document_declines_a_json_file(self, json_path: str) -> None:
        """Control: with the stock default, ``.json`` stays a structured suffix ReadDocument skips."""
        data_access = DataAccessCollection(files={"rod_payload": json_path})

        assert _RodStockJsonReadDocument.match_subclass_data_access(data_access, ["content"], Options()) is None

    def test_declared_default_makes_read_document_claim_json(self, json_path: str) -> None:
        """The declared ``frozenset({".json"})`` default claims ``.json`` with no option set."""
        data_access = DataAccessCollection(files={"rod_payload": json_path})

        matched = _RodJsonClaimingReadDocument.match_subclass_data_access(data_access, ["content"], Options())
        assert matched == json_path

    def test_explicit_option_still_overrides_the_read_document_default(self, json_path: str) -> None:
        """A user-set ``document_suffixes`` wins over the declared default."""
        data_access = DataAccessCollection(files={"rod_payload": json_path})
        options = Options({"document_suffixes": frozenset({".json"})})

        matched = _RodStockJsonReadDocument.match_subclass_data_access(data_access, ["content"], options)
        assert matched == json_path


class TestLocalReadersStayOutOfDiscovery:
    """None of the readers defined here can hijack reader selection elsewhere."""

    def test_no_local_reader_is_a_final_reader(self) -> None:
        for reader in (
            _RodFileProbe,
            _RodStockJsonReadFile,
            _RodJsonExcludingReadFile,
            _RodStockJsonReadDocument,
            _RodJsonClaimingReadDocument,
        ):
            assert reader.is_final_reader() is False
