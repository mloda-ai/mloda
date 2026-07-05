"""RED tests pinning the REVERT of the PyArrowReadFile intermediate base.

PR #559 introduced a ``PyArrowReadFile`` intermediate base and moved the five
structured file readers (CsvReader, JsonReader, ParquetReader, OrcReader,
FeatherReader) onto it, reframing ``ReadFile`` around a ``produce_table`` /
``check_backend`` template. That direction wrongly couples ``ReadFile`` to
pyarrow and is being reverted.

The target state these tests pin (they FAIL today, since PyArrowReadFile still
exists and the readers still subclass it):

  1. ``mloda_plugins.feature_group.input_data.pyarrow_read_file`` no longer
     exists: importing it raises ModuleNotFoundError and its symbol is gone.
  2. Each of the five readers is a DIRECT subclass of ``ReadFile`` and no
     ``PyArrowReadFile`` appears anywhere in its MRO (checked by name, since the
     class will not exist). Each reader overrides ``load_data`` wholesale again
     (its pre-#559 shape) instead of inheriting a produce_table template.
  3. ``ReadFile`` no longer defines ``produce_table`` or ``check_backend``.
  4. Structural classification still works: every reader is a final reader,
     ``ReadFile`` itself is not, and ``BaseInputData.supports_scoped_data_access``
     stays removed.
  5. Behavior is preserved end to end: reading a CSV still resolves through
     ``mloda.run_all`` into both PyArrowTable and PandasDataFrame.

The KEPT #559 improvements (``is_final_reader`` structural classification, the
hoisted ``init_reader``, the ReadDocument seam) are out of scope here and are
guarded by their own sibling test files.

No custom reader subclasses are defined at any scope, so there is nothing to
leak into the global plugin registry or the multiprocessing runners; the only
readers exercised are the five real, module-scope, in-repo readers.
"""

import importlib
import importlib.util
import os
from typing import Any

import pytest

from mloda.provider import BaseInputData
from mloda.user import Feature
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature  # noqa: F401
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader
from mloda_plugins.feature_group.input_data.read_files.json import JsonReader
from mloda_plugins.feature_group.input_data.read_files.orc import OrcReader
from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader

_PYARROW_READ_FILE_MODULE = "mloda_plugins.feature_group.input_data.pyarrow_read_file"

_READERS: list[type[ReadFile]] = [CsvReader, JsonReader, ParquetReader, OrcReader, FeatherReader]


class TestPyArrowReadFileBaseIsRemoved:
    """The intermediate pyarrow base and its module must be gone."""

    def test_pyarrow_read_file_module_cannot_be_imported(self) -> None:
        """Importing the module raises ModuleNotFoundError once #559's base is reverted.

        Fails today: the module still exists and imports cleanly.
        """
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(_PYARROW_READ_FILE_MODULE)

    def test_pyarrow_read_file_module_spec_is_gone(self) -> None:
        """find_spec returns None for a module that no longer exists.

        Fails today: find_spec still resolves a real spec for the module.
        """
        assert importlib.util.find_spec(_PYARROW_READ_FILE_MODULE) is None


class TestReadersAreDirectSubclassesOfReadFile:
    """Each reader reverts to ``class XReader(ReadFile)`` with no pyarrow base."""

    @pytest.mark.parametrize("reader", _READERS)
    def test_reader_is_direct_subclass_of_read_file(self, reader: type[ReadFile]) -> None:
        """ReadFile is the direct base and PyArrowReadFile is nowhere in the MRO.

        Fails today: the direct base is PyArrowReadFile, which also sits in the MRO.
        """
        assert issubclass(reader, ReadFile)
        assert ReadFile in reader.__bases__, f"{reader.__name__} must subclass ReadFile directly"

        mro_names = [klass.__name__ for klass in reader.__mro__]
        assert "PyArrowReadFile" not in mro_names, f"{reader.__name__} must not inherit from any PyArrowReadFile"

    @pytest.mark.parametrize("reader", _READERS)
    def test_reader_overrides_load_data_wholesale(self, reader: type[ReadFile]) -> None:
        """Each reader overrides load_data again (pre-#559 shape), not a produce_table template.

        Fails today: readers inherit the produce_table-based load_data template and
        do not override load_data relative to ReadFile.
        """
        assert reader._is_overridden(ReadFile, "load_data"), (
            f"{reader.__name__} must override load_data wholesale to return its table"
        )


class TestReadFileIsBackendNeutralAgain:
    """ReadFile sheds the produce_table / check_backend template machinery."""

    def test_read_file_has_no_produce_table(self) -> None:
        """Fails today: ReadFile still defines the produce_table read hook."""
        assert not hasattr(ReadFile, "produce_table")

    def test_read_file_has_no_check_backend(self) -> None:
        """Fails today: ReadFile still defines the check_backend guard hook."""
        assert not hasattr(ReadFile, "check_backend")


class TestClassificationStillWorks:
    """Structural final-reader classification survives the revert."""

    @pytest.mark.parametrize("reader", _READERS)
    def test_reader_classifies_as_final(self, reader: type[ReadFile]) -> None:
        assert reader.is_final_reader() is True

    def test_read_file_base_is_not_final(self) -> None:
        assert ReadFile.is_final_reader() is False

    def test_supports_scoped_data_access_stays_removed(self) -> None:
        assert not hasattr(BaseInputData, "supports_scoped_data_access")


class TestCsvReadStillWorksEndToEnd:
    """Behavior parity: a CSV still resolves through run_all after the revert."""

    file_path = f"{os.getcwd()}/tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"
    feature_list: list[str] = ["id", "V1", "V2"]

    def _features(self) -> list[Any]:
        features: list[Any] = []
        for name in self.feature_list:
            feature = Feature(name=name)
            feature.options.add_to_group(CsvReader.__name__, self.file_path)
            features.append(feature)
        return features

    def test_csv_reads_into_pyarrow_table(self) -> None:
        result = mloda.run_all(self._features(), compute_frameworks=["PyArrowTable"])
        columns = result[0].to_pydict()
        for name in self.feature_list:
            assert name in columns

    def test_csv_reads_into_pandas_dataframe(self) -> None:
        result = mloda.run_all(self._features(), compute_frameworks=["PandasDataFrame"])
        columns = list(result[0].columns)
        for name in self.feature_list:
            assert name in columns
