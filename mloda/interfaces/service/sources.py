"""The configured sources as mloda sees them: their files, readers, columns and data access collection."""

import inspect
import os
from collections.abc import Mapping
from functools import partial

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.utils import safe_field
from mloda.interfaces.service.config import SourceConfig
from mloda.provider import ReadFileFG, get_all_subclasses

# A file format group signals "cannot tell" by raising one of these.
READER_ERRORS: tuple[type[Exception], ...] = (ImportError, OSError, ValueError)


def readers_for(path: str) -> list[type[ReadFileFG]]:
    """The loaded concrete ReadFileFG classes that own ``path`` by suffix, in a stable order."""
    groups = [fg for fg in get_all_subclasses(ReadFileFG) if not inspect.isabstract(fg)]
    matching = [fg for fg in groups if path.endswith(fg.default_owned_suffixes())]
    return sorted(matching, key=lambda fg: fg.__qualname__)


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
            columns.update(safe_field(partial(reader.column_names, path), (), catching=READER_ERRORS))
    return sorted(columns)


def data_access_collection(sources: Mapping[str, SourceConfig]) -> DataAccessCollection:
    """A DataAccessCollection over the files and folders of ``sources``."""
    files = sorted({file for source in sources.values() for file in source.files})
    folders = sorted({folder for source in sources.values() for folder in source.folders})
    return DataAccessCollection(files=files, folders=folders)
