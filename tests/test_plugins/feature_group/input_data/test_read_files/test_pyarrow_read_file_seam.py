"""Tests for the planned ReadFile / PyArrowReadFile backend split.

These pin an intentionally BREAKING refactor of the ReadFile lifecycle seam:

    ReadFile (mloda_plugins/feature_group/input_data/read_file.py) becomes
    backend-neutral:
      - _pyarrow_module, check_pyarrow_backend, and _file_format_label are
        REMOVED from ReadFile entirely.
      - classmethod check_backend(cls) -> None is ADDED: the default is a
        documented no-op returning None; a backend family overrides it to
        verify its engine is importable before any read happens.
      - load_data keeps the template order: probe produce_table (raise
        NotImplementedError when not overridden), then cls.check_backend(),
        then cls.produce_table(data_access, list(features.get_all_names())).
      - _final_reader_requires() == ("produce_table", "suffix"): a reader
        overriding only produce_table plus suffix is a final reader with no
        pyarrow knowledge at all.

    A NEW module mloda_plugins/feature_group/input_data/pyarrow_read_file.py
    defines PyArrowReadFile(ReadFile), the intermediate base carrying all
    pyarrow knowledge:
      - classmethod _pyarrow_module(cls) -> Any: default raises
        NotImplementedError; a reader returns its pyarrow submodule, or None
        when pyarrow is absent.
      - class attribute _file_format_label: str, used in the guard message.
      - check_backend override: raises
        ImportError(f"pyarrow is required to read {label} files. Install it
        with: pip install 'mloda[pyarrow]'") when _pyarrow_module() is None.
      - redeclares _final_reader_requires() ==
        ("produce_table", "suffix", "_pyarrow_module"), moving the
        classification anchor to PyArrowReadFile for its subtree.

    The five in-repo readers (CsvReader, JsonReader, ParquetReader, OrcReader,
    FeatherReader) subclass PyArrowReadFile instead of ReadFile, keep their
    _file_format_label values, and keep the inherited load_data template.

Red-phase mechanics:
PyArrowReadFile does not exist yet, so it is resolved dynamically; the classes
in the PyArrowReadFile family below fall back to ReadFile as their base until
the module lands, and every test touching the family first asserts (with a
clear message) that the planned base exists and is the actual base. This keeps
the module importable and type-clean while still failing for planned-contract
reasons only.

Test isolation note:
All reader subclasses used here are defined at MODULE scope, never inside test
methods. Function-local subclasses linger in the global plugin registry until
GC runs and are unpicklable, which breaks multiprocessing runners. Every
suffix is a unique ``.zzzpaseam*`` sentinel that matches no real file in this
repo or in any sibling test; ReadFile matching is suffix-based, so a leaked
registry entry can never match a real file. ``get_column_names`` stays
unimplemented so ``validate_columns`` defaults permissive, exactly like the
sibling seam test files.
"""

import importlib
import importlib.util
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

_PYARROW_READ_FILE_MODULE = "mloda_plugins.feature_group.input_data.pyarrow_read_file"


def _underlying(member: Any) -> Any:
    """Underlying function of a classmethod/staticmethod/plain override, for identity comparison."""
    return getattr(member, "__func__", member)


def _planned(owner: type, name: str) -> Any:
    """Fetch a planned attribute, failing with a clear message while it does not exist yet."""
    member = getattr(owner, name, None)
    assert member is not None, f"{owner.__name__} must define {name} (planned ReadFile/PyArrowReadFile split)"
    return member


def _load_pyarrow_read_file() -> "type[ReadFile] | None":
    """Resolve the planned PyArrowReadFile class, or None while its module does not exist yet."""
    if importlib.util.find_spec(_PYARROW_READ_FILE_MODULE) is None:
        return None
    module = importlib.import_module(_PYARROW_READ_FILE_MODULE)
    loaded: type[ReadFile] = module.PyArrowReadFile
    return loaded


_PYARROW_READ_FILE: "type[ReadFile] | None" = _load_pyarrow_read_file()

# Fallback base keeps this module importable (and its module-scope classes harmless) until the
# planned module lands; tests never rely on the fallback, they assert the real base first.
_PyArrowSeamBase: type[ReadFile] = ReadFile if _PYARROW_READ_FILE is None else _PYARROW_READ_FILE


def _pyarrow_read_file() -> type[ReadFile]:
    """The planned PyArrowReadFile base; fails with a clear red-phase message while missing."""
    assert _PYARROW_READ_FILE is not None, (
        f"{_PYARROW_READ_FILE_MODULE} must define PyArrowReadFile (planned backend split contract)"
    )
    return _PYARROW_READ_FILE


# --------------------------------------------------------------------------------------
# Module-scope readers. The first subclasses backend-neutral ReadFile directly; the
# _Pa* family targets PyArrowReadFile (dynamic base, see red-phase mechanics above).
# --------------------------------------------------------------------------------------


class _BackendNeutralSeamFile(ReadFile):
    """Backend-neutral final reader: ONLY produce_table + sentinel suffix, no pyarrow method at all."""

    sentinel_table: ClassVar[dict[str, str]] = {"table": "backend-neutral"}
    produce_calls: ClassVar[list[tuple[Any, list[str]]]] = []

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzpaseamneutral",)

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        cls.produce_calls.append((data_access, column_names))
        return cls.sentinel_table


class _PaTableSuffixOnlyFile(_PyArrowSeamBase):  # type: ignore[misc, valid-type]
    """PyArrowReadFile child overriding produce_table + sentinel suffix but NOT _pyarrow_module: not final."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzpaseamonly",)

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        return {"table": "pa-only"}


class _PaCompleteFile(_PyArrowSeamBase):  # type: ignore[misc, valid-type]
    """PyArrowReadFile child with the full hook set; _pyarrow_module returns a truthy sentinel."""

    backend_sentinel: ClassVar[object] = object()

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzpaseamfull",)

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        return {"table": "pa-full"}

    @classmethod
    def _pyarrow_module(cls) -> Any:
        return cls.backend_sentinel


class _PaNoBackendFile(_PyArrowSeamBase):  # type: ignore[misc, valid-type]
    """PyArrowReadFile child whose _pyarrow_module returns None: the check_backend guard must fire."""

    _file_format_label = "ZzzPaSeamFormat"

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzpaseamnobe",)

    @classmethod
    def _pyarrow_module(cls) -> Any:
        return None

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        raise AssertionError("produce_table must not run when the pyarrow backend is missing")


class TestReadFileIsBackendNeutral:
    """ReadFile must lose all pyarrow knowledge (planned breaking change)."""

    def test_pyarrow_names_are_gone_from_read_file(self) -> None:
        """The three pyarrow names move to PyArrowReadFile; ReadFile keeps none of them.

        Fails today: ReadFile still defines _pyarrow_module, check_pyarrow_backend,
        and _file_format_label.
        """
        assert "_pyarrow_module" not in ReadFile.__dict__
        assert "check_pyarrow_backend" not in ReadFile.__dict__
        assert "_file_format_label" not in ReadFile.__dict__
        assert not hasattr(ReadFile, "_pyarrow_module")
        assert not hasattr(ReadFile, "check_pyarrow_backend")

    def test_final_reader_requires_drops_pyarrow_module(self) -> None:
        """Backend-neutral ReadFile requires only the read hook and suffix.

        Fails today: the tuple still includes _pyarrow_module.
        """
        assert ReadFile._final_reader_requires() == ("produce_table", "suffix")

    def test_check_backend_default_is_noop(self) -> None:
        """ReadFile.check_backend is the backend-neutral pre-read guard hook; the default is a no-op.

        Fails today: ReadFile has no check_backend classmethod at all.
        """
        assert "check_backend" in ReadFile.__dict__, "ReadFile must define check_backend (planned guard hook)"
        assert _planned(ReadFile, "check_backend")() is None

    def test_backend_neutral_reader_is_final(self) -> None:
        """produce_table + suffix with NO pyarrow method must classify as a final reader.

        Fails today: ReadFile's requires-tuple still demands _pyarrow_module.
        """
        assert _BackendNeutralSeamFile.is_final_reader() is True

    def test_backend_neutral_reader_load_data_runs_end_to_end(self) -> None:
        """load_data runs probe -> check_backend (no-op) -> produce_table without any pyarrow hook.

        Fails today: check_pyarrow_backend calls the abstract _pyarrow_module and
        raises NotImplementedError before produce_table can run.
        """
        _BackendNeutralSeamFile.produce_calls.clear()

        features = FeatureSet()
        features.add(Feature("col_x"))
        features.add(Feature("col_y"))

        sentinel_access = object()
        result = _BackendNeutralSeamFile.load_data(sentinel_access, features)

        assert result is _BackendNeutralSeamFile.sentinel_table
        assert len(_BackendNeutralSeamFile.produce_calls) == 1
        recorded_access, recorded_columns = _BackendNeutralSeamFile.produce_calls[-1]
        assert recorded_access is sentinel_access, "produce_table must receive the original data_access"
        assert recorded_columns == list(features.get_all_names())
        assert sorted(recorded_columns) == ["col_x", "col_y"]


class TestPyArrowReadFileBase:
    """PyArrowReadFile: the new intermediate base carrying the pyarrow knowledge.

    All of these fail today: the pyarrow_read_file module does not exist yet.
    """

    def test_exists_and_subclasses_read_file(self) -> None:
        base = _pyarrow_read_file()
        assert issubclass(base, ReadFile)

    def test_not_final_itself(self) -> None:
        """No hooks overridden on the base itself: it must stay out of plugin discovery."""
        base = _pyarrow_read_file()
        assert base.is_final_reader() is False

    def test_redeclares_final_reader_requires_as_anchor(self) -> None:
        """PyArrowReadFile redeclares the tuple in its own __dict__, becoming the subtree anchor."""
        base = _pyarrow_read_file()
        assert "_final_reader_requires" in base.__dict__
        assert base._final_reader_requires() == ("produce_table", "suffix", "_pyarrow_module")

    def test_pyarrow_module_default_raises_not_implemented(self) -> None:
        base = _pyarrow_read_file()
        with pytest.raises(NotImplementedError):
            _planned(base, "_pyarrow_module")()

    def test_overrides_check_backend_and_declares_label(self) -> None:
        base = _pyarrow_read_file()
        assert _underlying(_planned(base, "check_backend")) is not _underlying(_planned(ReadFile, "check_backend"))
        assert isinstance(_planned(base, "_file_format_label"), str)


class TestPyArrowAnchorSemantics:
    """_final_reader_requires anchors at PyArrowReadFile for its subtree.

    All of these fail today: the pyarrow_read_file module does not exist yet,
    so the base-existence assert fires first.
    """

    def test_produce_table_and_suffix_without_pyarrow_module_is_not_final(self) -> None:
        """On the pyarrow subtree the backend hook stays required, unlike on neutral ReadFile."""
        assert issubclass(_PaTableSuffixOnlyFile, _pyarrow_read_file())
        assert _PaTableSuffixOnlyFile.is_final_reader() is False

    def test_adding_pyarrow_module_makes_it_final(self) -> None:
        assert issubclass(_PaCompleteFile, _pyarrow_read_file())
        assert _PaCompleteFile.is_final_reader() is True

    def test_check_backend_guard_fires_before_features_access(self, tmp_path: Path) -> None:
        """load_data(target, None) raises the exact centralized ImportError before touching features."""
        assert issubclass(_PaNoBackendFile, _pyarrow_read_file())
        target = str(tmp_path / "data.zzzpaseamnobe")

        with pytest.raises(ImportError) as excinfo:
            _PaNoBackendFile.load_data(target, None)

        assert str(excinfo.value) == (
            "pyarrow is required to read ZzzPaSeamFormat files. Install it with: pip install 'mloda[pyarrow]'"
        )


class TestInRepoReadersMigrateToPyArrowReadFile:
    """The five concrete readers move onto the pyarrow intermediate base.

    All of these fail today: the pyarrow_read_file module does not exist yet,
    so the base-existence assert fires first.
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
    def test_reader_subclasses_pyarrow_read_file(self, reader: type[ReadFile], label: str) -> None:
        base = _pyarrow_read_file()
        assert issubclass(reader, base), f"{reader.__name__} must subclass PyArrowReadFile"
        assert reader.is_final_reader() is True
        assert _planned(reader, "_file_format_label") == label
        assert _underlying(reader.load_data) is _underlying(ReadFile.load_data), (
            f"{reader.__name__} must keep the inherited load_data template"
        )
        assert _planned(reader, "_pyarrow_module")() is not None, (
            f"{reader.__name__}._pyarrow_module() must return its pyarrow submodule (pyarrow is installed here)"
        )
