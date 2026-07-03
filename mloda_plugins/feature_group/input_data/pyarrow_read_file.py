from typing import Any

from mloda_plugins.feature_group.input_data.read_file import ReadFile


class PyArrowReadFile(ReadFile):
    """
    Intermediate base for pyarrow-backed file readers.

    A concrete reader implements ``produce_table``, ``suffix``, and
    ``_pyarrow_module`` (plus ``_file_format_label`` for the guard message).
    """

    _file_format_label: str = "structured"

    @classmethod
    def _pyarrow_module(cls) -> Any:
        """Return this reader's pyarrow submodule, or None when pyarrow is absent."""
        raise NotImplementedError

    @classmethod
    def check_backend(cls) -> None:
        """Raises the centralized ImportError when the reader's pyarrow backend is missing."""
        if cls._pyarrow_module() is None:
            raise ImportError(
                f"pyarrow is required to read {cls._file_format_label} files. "
                "Install it with: pip install 'mloda[pyarrow]'"
            )

    @classmethod
    def _final_reader_requires(cls) -> tuple[str, ...]:
        # The classification anchor moves here for the pyarrow subtree.
        return ("produce_table", "suffix", "_pyarrow_module")
