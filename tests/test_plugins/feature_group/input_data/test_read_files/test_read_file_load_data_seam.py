"""Tests for the ReadFile.load_data template-method lifecycle seam.

These pin the planned contract (mirroring the ReadDB seam from issue #535 and
the ReadDocument seam):

    ReadFile gains
      - classmethod produce_table(cls, data_access, column_names) -> Any: the
        per-format read hook; the base default raises NotImplementedError.
      - classmethod _pyarrow_module(cls) -> Any: the format's pyarrow submodule,
        or None when pyarrow is absent; the base default raises NotImplementedError.
      - class attribute _file_format_label: str (e.g. "CSV", "Parquet") used in
        the centralized guard message.
      - classmethod check_pyarrow_backend(cls) -> None: raises
        ImportError(f"pyarrow is required to read {label} files. Install it "
        f"with: pip install 'mloda[pyarrow]'") when _pyarrow_module() returns
        None; no-op otherwise.

    load_data becomes a template method with a STRICT order:
      1. probe the read hook FIRST: if produce_table is still the ReadFile base
         default, raise NotImplementedError IMMEDIATELY (before any backend
         check, so a hook-less _pyarrow_module recorder must stay empty),
      2. check_pyarrow_backend(): the ImportError fires BEFORE features access,
         so load_data(anything, None) on a backend-less seam reader raises
         ImportError, never AttributeError (this preserves the contract in
         tests/test_core/test_optional_pyarrow/test_file_readers_without_pyarrow.py,
         where CsvReader.load_data("/nonexistent/path.csv", None) must raise
         ImportError mentioning pyarrow under blocked pyarrow),
      3. return cls.produce_table(data_access, list(features.get_all_names())).

    supports_scoped_data_access() is decided STRUCTURALLY (no execution):
    True iff load_data is overridden wholesale relative to ReadFile, or
    produce_table is overridden. ReadFile itself and hook-less intermediate
    bases stay False.

Test isolation note:
All ReadFile subclasses used by these tests are defined at MODULE scope, never
inside test methods. Function-local subclasses linger in
``ReadFile.__subclasses__()`` (the global plugin registry) until GC runs,
leaking into sibling tests' plugin discovery, and they are unpicklable, which
breaks multiprocessing runners. Module-scope classes are picklable and stable.
ReadFile matching is suffix-based, so every seam reader uses a sentinel suffix
in the ``.zzzfileseam*`` family that no real file in this repo or in any
sibling test has; a leaked registry entry can therefore never match a file in
``test_read_file.py`` or the multiprocessing reader tests. ``get_column_names``
is left unimplemented on purpose so ``validate_columns`` defaults permissive,
keeping leaked classes harmless exactly like the ReadDB/ReadDocument seam
test files.
"""

from pathlib import Path
from typing import Any, ClassVar

import pytest

from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader
from mloda_plugins.feature_group.input_data.read_files.json import JsonReader
from mloda_plugins.feature_group.input_data.read_files.orc import OrcReader
from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader


def _underlying(member: Any) -> Any:
    """Underlying function of a classmethod/staticmethod/plain override, for identity comparison."""
    return getattr(member, "__func__", member)


def _planned(owner: type, name: str) -> Any:
    """Fetch a planned seam attribute, failing with a clear message while it does not exist yet."""
    member = getattr(owner, name, None)
    assert member is not None, f"{owner.__name__} must define {name} (planned ReadFile seam contract)"
    return member


# --------------------------------------------------------------------------------------
# Module-scope readers for the load_data lifecycle seam tests.
#
# Defined at MODULE scope (never inside a test) so they are picklable and stable in the
# plugin registry. Every suffix is a unique ``.zzzfileseam*`` sentinel that matches no
# real file anywhere, so a leaked class can never be selected by sibling tests' plugin
# discovery (ReadFile matching is suffix-based). The tests below drive these readers
# directly and record interactions via ClassVar lists.
# --------------------------------------------------------------------------------------


class _SeamIntermediateFile(ReadFile):
    """Intermediate base: overrides suffix() and _pyarrow_module but NOT produce_table; not final."""

    pyarrow_module_calls: ClassVar[list[str]] = []

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzfileseamint",)

    @classmethod
    def _pyarrow_module(cls) -> Any:
        cls.pyarrow_module_calls.append("called")
        return object()


class _SeamHappyPathFile(ReadFile):
    """Final seam reader: truthy backend, produce_table records its arguments and returns a sentinel."""

    backend_sentinel: ClassVar[object] = object()
    sentinel_table: ClassVar[dict[str, str]] = {"table": "sentinel"}
    produce_calls: ClassVar[list[tuple[Any, list[str]]]] = []

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzfileseam",)

    @classmethod
    def _pyarrow_module(cls) -> Any:
        return cls.backend_sentinel

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        cls.produce_calls.append((data_access, column_names))
        return cls.sentinel_table


class _SeamNoBackendFile(ReadFile):
    """Final seam reader whose backend is absent: the centralized guard must fire before features access."""

    _file_format_label = "ZzzSeamFormat"

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzfileseamnb",)

    @classmethod
    def _pyarrow_module(cls) -> Any:
        return None

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        raise AssertionError("produce_table must not run when the pyarrow backend is missing")


class _ProbeRecorderFile(ReadFile):
    """Final seam reader with recorders on both hooks; the probe must invoke neither."""

    pyarrow_module_calls: ClassVar[list[str]] = []
    produce_calls: ClassVar[list[tuple[Any, list[str]]]] = []

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzfileseamprobe",)

    @classmethod
    def _pyarrow_module(cls) -> Any:
        cls.pyarrow_module_calls.append("called")
        return object()

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        cls.produce_calls.append((data_access, column_names))
        return "recorded"


class _BoomProbeFile(ReadFile):
    """Final seam reader whose read hook explodes; the probe must never invoke it."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzfileseamboom",)

    @classmethod
    def _pyarrow_module(cls) -> Any:
        return object()

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        raise RuntimeError("boom")


class _WholesaleFile(ReadFile):
    """Legacy-style reader: overrides load_data wholesale, no produce_table; still final."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzfileseamwhole",)

    @classmethod
    def load_data(cls, data_access: Any, features: Any) -> Any:
        return [{"_WholesaleFile": "wholesale"}]


class TestReadFileLoadDataSeam:
    def test_abstract_read_file_probe_stays_false(self) -> None:
        """Abstract ReadFile has no read hook: load_data raises and probe is False."""
        with pytest.raises(NotImplementedError):
            ReadFile.load_data(None, None)  # type: ignore[arg-type]

        assert ReadFile.supports_scoped_data_access() is False

    def test_intermediate_base_probe_false_and_raises_before_backend_guard(self) -> None:
        """A base that overrides suffix() and _pyarrow_module but NOT produce_table is not final.

        The read hook must be probed *before* the backend guard, so
        load_data(None, None) raises NotImplementedError and the
        _pyarrow_module recorder must show the guard never ran.
        """
        _SeamIntermediateFile.pyarrow_module_calls.clear()

        assert _SeamIntermediateFile.supports_scoped_data_access() is False

        with pytest.raises(NotImplementedError):
            _SeamIntermediateFile.load_data(None, None)  # type: ignore[arg-type]

        assert _SeamIntermediateFile.pyarrow_module_calls == [], (
            "_pyarrow_module() must not run before the read hook is probed"
        )

    def test_seam_happy_path_classifies_final(self) -> None:
        """A reader overriding produce_table is final.

        Fails today: there is no produce_table seam, so the base probe runs
        load_data(None, None), hits ReadFile's NotImplementedError, and
        wrongly classifies the reader as False.
        """
        assert _SeamHappyPathFile.supports_scoped_data_access() is True

    def test_seam_happy_path_load_data_delegates_to_produce_table(self) -> None:
        """load_data probes the hook, checks the backend, then delegates to produce_table.

        produce_table must receive the ORIGINAL data_access and exactly
        list(features.get_all_names()).
        """
        _SeamHappyPathFile.produce_calls.clear()

        features = FeatureSet()
        features.add(Feature("col_a"))
        features.add(Feature("col_b"))

        sentinel_access = object()
        result = _SeamHappyPathFile.load_data(sentinel_access, features)

        assert result is _SeamHappyPathFile.sentinel_table
        assert len(_SeamHappyPathFile.produce_calls) == 1
        recorded_access, recorded_columns = _SeamHappyPathFile.produce_calls[-1]
        assert recorded_access is sentinel_access, "produce_table must receive the original data_access"
        assert recorded_columns == list(features.get_all_names())
        assert sorted(recorded_columns) == ["col_a", "col_b"]

    def test_backend_guard_fires_before_features_access(self, tmp_path: Path) -> None:
        """A seam reader without a pyarrow backend raises the centralized ImportError.

        features=None proves the guard fires BEFORE features access (this is the
        ordering the blocked-pyarrow contract in
        test_file_readers_without_pyarrow.py relies on).
        """
        target = str(tmp_path / "data.zzzfileseamnb")

        with pytest.raises(ImportError) as excinfo:
            _SeamNoBackendFile.load_data(target, None)  # type: ignore[arg-type]

        message = str(excinfo.value)
        assert "pyarrow is required to read" in message
        assert "ZzzSeamFormat" in message
        assert "pip install 'mloda[pyarrow]'" in message

    def test_check_pyarrow_backend_raises_without_module(self) -> None:
        """check_pyarrow_backend() alone raises the same centralized ImportError.

        Fails today: ReadFile has no check_pyarrow_backend classmethod at all.
        """
        guard = _planned(_SeamNoBackendFile, "check_pyarrow_backend")

        with pytest.raises(ImportError) as excinfo:
            guard()

        message = str(excinfo.value)
        assert "pyarrow is required to read" in message
        assert "ZzzSeamFormat" in message
        assert "pip install 'mloda[pyarrow]'" in message

    def test_check_pyarrow_backend_noop_with_truthy_module(self) -> None:
        """check_pyarrow_backend() returns None when _pyarrow_module() is truthy."""
        guard = _planned(_SeamHappyPathFile, "check_pyarrow_backend")

        assert guard() is None


class TestSupportsScopedDataAccessIsSideEffectFree:
    """supports_scoped_data_access() must classify readers structurally, not by execution."""

    def test_probe_does_not_run_pyarrow_module_or_produce_table(self) -> None:
        """A seam reader is final, but probing it must NOT execute any hook.

        Fails today: there is no produce_table seam, so the base probe runs
        load_data(None, None), hits ReadFile's NotImplementedError, and
        wrongly classifies the reader as False.
        """
        _ProbeRecorderFile.pyarrow_module_calls.clear()
        _ProbeRecorderFile.produce_calls.clear()

        is_final = _ProbeRecorderFile.supports_scoped_data_access()

        assert is_final is True
        assert _ProbeRecorderFile.pyarrow_module_calls == [], "probe must not call _pyarrow_module()"
        assert _ProbeRecorderFile.produce_calls == [], "probe must not call produce_table()"

    def test_probe_does_not_raise_when_produce_table_would_raise(self) -> None:
        """Probing a reader whose read hook raises must still return True, not propagate."""
        is_final = _BoomProbeFile.supports_scoped_data_access()

        assert is_final is True

    def test_wholesale_load_data_override_is_final(self) -> None:
        """Regression guard: overriding load_data wholesale still counts as a final reader."""
        assert _WholesaleFile.supports_scoped_data_access() is True

    def test_abstract_and_intermediate_bases_stay_false(self) -> None:
        """Regression guard: classes without a read hook are not final readers."""
        assert ReadFile.supports_scoped_data_access() is False
        assert _SeamIntermediateFile.supports_scoped_data_access() is False


class TestInRepoReadersMigrateToTheSeam:
    """The five concrete readers must adopt the seam instead of rewriting load_data.

    All of these fail today: each reader still overrides load_data wholesale
    (repeating the pyarrow guard and column selection) and none defines
    produce_table, _pyarrow_module, or _file_format_label.
    """

    @pytest.mark.parametrize(
        ("reader", "label"),
        [
            (CsvReader, "CSV"),
            (JsonReader, "JSON"),
            (ParquetReader, "Parquet"),
            (OrcReader, "ORC"),
            (FeatherReader, "Feather"),
        ],
    )
    def test_readers_use_inherited_load_data_template(self, reader: type[ReadFile], label: str) -> None:
        """Each in-repo reader inherits ReadFile.load_data and overrides the seam hooks."""
        assert _underlying(reader.load_data) is _underlying(ReadFile.load_data), (
            f"{reader.__name__} must not override load_data wholesale anymore"
        )

        base_hook = _planned(ReadFile, "produce_table")
        reader_hook = _planned(reader, "produce_table")
        assert _underlying(reader_hook) is not _underlying(base_hook), (
            f"{reader.__name__} must override the produce_table read hook"
        )

        module = _planned(reader, "_pyarrow_module")()
        assert module is not None, f"{reader.__name__}._pyarrow_module() must return the module (pyarrow is installed)"

        assert _planned(reader, "_file_format_label") == label


class TestCsvReaderEndToEndParity:
    """Guards the migration: the representative CSV reader keeps its observable behavior."""

    def test_csv_reader_returns_exactly_the_requested_columns(self, tmp_path: Path) -> None:
        """CsvReader.load_data selects exactly the FeatureSet's columns from the file."""
        file_path = tmp_path / "parity.csv"
        file_path.write_text("id,alpha,beta\n1,10,100\n2,20,200\n", encoding="utf-8")

        features = FeatureSet()
        features.add(Feature("id"))
        features.add(Feature("beta"))

        table = CsvReader.load_data(str(file_path), features)

        assert sorted(table.column_names) == ["beta", "id"]
        assert table.column("beta").to_pylist() == [100, 200]
        assert table.column("id").to_pylist() == [1, 2]


# --------------------------------------------------------------------------------------
# Module-scope readers pinning the review-gap fix: a partial hook set is NOT final.
#
# Two reviews converged on the same hole: supports_scoped_data_access() classifies a
# reader as final from the produce_table override alone. An intermediate base that
# overrides produce_table but leaves suffix() (or _pyarrow_module()) abstract then
# enters plugin discovery as a final scoped reader, and match_subclass_data_access ->
# _file_matches calls the abstract suffix(), raising NotImplementedError and crashing
# file resolution. Mirroring the ReadDB family pattern (_RowHookNoConnectDB requires
# BOTH produce_rows AND connect), a ReadFile subclass is "final via the seam" only if
# produce_table AND suffix AND _pyarrow_module are ALL overridden relative to ReadFile.
#
# The suffix-less classes below are exactly the hazardous shape, so each one overrides
# match_subclass_data_access to return None: even while misclassified as final (today,
# before the fix), a leaked registry entry can never be probed into the abstract
# suffix() by a sibling test's file resolution. supports_scoped_data_access() is
# structural and never consults match_subclass_data_access, so the guard does not
# distort any assertion below.
# --------------------------------------------------------------------------------------


class _TableHookNoSuffixFile(ReadFile):
    """Overrides produce_table and _pyarrow_module but NOT suffix: cannot match files, so not final.

    A reader with an abstract suffix() would crash _file_matches during discovery, so the
    classification screen must reject it (like _RowHookNoConnectDB in the ReadDB seam tests).
    """

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        return {"table": "no-suffix"}

    @classmethod
    def _pyarrow_module(cls) -> Any:
        return object()

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Any) -> Any:
        return None


class _TableHookNoBackendFile(ReadFile):
    """Overrides produce_table and suffix (sentinel) but NOT _pyarrow_module: not final.

    Without a _pyarrow_module override the centralized backend guard itself is abstract,
    so load_data cannot even decide whether the backend exists; the classification screen
    must reject the class. The sentinel suffix matches no real file anywhere.
    """

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzfileseamnopa",)

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        return {"table": "no-backend"}


class _TableHookNoSuffixBaseFile(ReadFile):
    """Intermediate base: overrides produce_table and _pyarrow_module, leaves suffix abstract.

    This is the exact reviewer-reported shape: a shared backend base for a family of
    formats. It must classify as NOT final; only concrete children that add suffix are.
    """

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        return {"table": "base"}

    @classmethod
    def _pyarrow_module(cls) -> Any:
        return object()

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Any) -> Any:
        return None


class _SuffixOnlyChildFile(_TableHookNoSuffixBaseFile):
    """Concrete child completing the base with only a sentinel suffix: the full hook set, final."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzfileseamsfxchild",)


class TestReadFileSeamClassificationScreensPartialHookSets:
    """Final via the seam iff produce_table AND suffix AND _pyarrow_module are all overridden."""

    def test_table_hook_without_suffix_is_not_final(self) -> None:
        """A read hook without a suffix override cannot match files, so it must be screened out.

        Fails today: the structural check returns True from the produce_table override
        alone, and discovery would then crash in _file_matches on the abstract suffix().
        """
        assert _TableHookNoSuffixFile.supports_scoped_data_access() is False

    def test_table_hook_without_pyarrow_module_is_not_final(self) -> None:
        """A read hook without a _pyarrow_module override has no backend guard, so it is not final.

        Fails today: the structural check ignores that _pyarrow_module is still the
        abstract base default, so load_data would raise NotImplementedError inside
        check_pyarrow_backend instead of running the seam.
        """
        assert _TableHookNoBackendFile.supports_scoped_data_access() is False

    def test_intermediate_backend_base_without_suffix_is_not_final(self) -> None:
        """An intermediate backend base (produce_table + _pyarrow_module, no suffix) is not final.

        Fails today: the produce_table override alone classifies the base as final, so it
        enters discovery and _file_matches calls the abstract suffix().
        """
        assert _TableHookNoSuffixBaseFile.supports_scoped_data_access() is False

    def test_concrete_child_completing_the_hook_set_is_final(self) -> None:
        """Regression guard: a child adding only suffix on top of a backend base stays final.

        Passes today; pins that tightening the screen must not reject inherited overrides.
        """
        assert _SuffixOnlyChildFile.supports_scoped_data_access() is True
