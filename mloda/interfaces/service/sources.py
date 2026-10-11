"""The configured sources as mloda sees them: their files, readers, columns and data access collection."""

import os
from collections.abc import Mapping
from functools import partial

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.utils import get_all_subclasses, safe_field
from mloda.interfaces.service.config import SourceConfig
from mloda_plugins.feature_group.input_data.read_file import ReadFile

# The reader contract signals "cannot tell" by raising one of these.
READER_ERRORS: tuple[type[Exception], ...] = (NotImplementedError, ImportError, OSError, ValueError)


def readers_for(path: str) -> list[type[ReadFile]]:
    """The ReadFile classes whose suffix matches ``path``, in a stable order."""
    readers = get_all_subclasses(ReadFile)
    matching = [r for r in readers if path.endswith(safe_field(r.suffix, (), catching=(NotImplementedError,)))]
    return sorted(matching, key=lambda reader: reader.__qualname__)


def source_files(sources: Mapping[str, SourceConfig]) -> list[str]:
    """Every existing file of ``sources``, including the direct entries of their folders."""
    files = [file for source in sources.values() for file in source.files]
    for folder in (folder for source in sources.values() for folder in source.folders):
        if os.path.isdir(folder):
            files.extend(os.path.join(folder, entry) for entry in sorted(os.listdir(folder)))
    return [file for file in dict.fromkeys(files) if os.path.isfile(file)]


def source_columns(sources: Mapping[str, SourceConfig]) -> list[str]:
    """The sorted column names any reader can describe in the files of ``sources``."""
    columns: set[str] = set()
    for path in source_files(sources):
        for reader in readers_for(path):
            columns.update(safe_field(partial(reader.describe_columns, path), {}, catching=READER_ERRORS))
    return sorted(columns)


def data_access_collection(sources: Mapping[str, SourceConfig]) -> DataAccessCollection:
    """A DataAccessCollection over the files and folders of ``sources``."""
    files = sorted({file for source in sources.values() for file in source.files})
    folders = sorted({folder for source in sources.values() for folder in source.folders})
    return DataAccessCollection(files=files, folders=folders)
