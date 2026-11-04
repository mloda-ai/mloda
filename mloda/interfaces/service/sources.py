"""The configured sources as mloda sees them: their files, readers, columns and data access collection."""

import inspect
import os
from collections.abc import Mapping
from functools import partial

from mloda.core.abstract_plugins.components.credential import Credential
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.utils import safe_field
from mloda.interfaces.service.config import ConfigError, CredentialRef, FileRef, Settings, SourceConfig
from mloda.provider import ReadFileFG, get_all_subclasses

# A ReadFileFG group signals "cannot tell" by raising one of these.
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


def _credential_value(handle: str, name: str, ref: CredentialRef, env: Mapping[str, str]) -> str:
    where = f"[mcp.credentials.{handle}] field '{name}'"
    if isinstance(ref, FileRef):
        if not os.path.isfile(ref.path) or not os.access(ref.path, os.R_OK):
            raise ConfigError(f"{where}: file {ref.path} is missing or not readable")
        with open(ref.path, encoding="utf-8") as stream:
            value = stream.read().removesuffix("\n")
        if not value:
            raise ConfigError(f"{where}: file {ref.path} is empty")
        return value
    value = env.get(ref, "")
    if not value:
        raise ConfigError(f"{where}: environment variable {ref} is unset or empty")
    return value


def credential_values(
    credentials: Mapping[str, Mapping[str, CredentialRef]], env: Mapping[str, str]
) -> dict[str, Credential]:
    """Read every configured credential from ``env`` and files; errors name the handle and field, never a value."""
    return {
        handle: Credential({name: _credential_value(handle, name, ref, env) for name, ref in fields.items()})
        for handle, fields in credentials.items()
    }


def check_credentials(settings: Settings, env: Mapping[str, str], base: DataAccessCollection | None = None) -> None:
    """Resolve every configured credential once and refuse a handle ``base`` already holds."""
    credential_values(settings.credentials, env)
    taken = sorted(set(settings.credentials) & set(base.handles() if base is not None else ()))
    if taken:
        raise ConfigError(f"[mcp.credentials.{taken[0]}]: the handle is already registered in the given collection")


def data_access_collection(
    sources: Mapping[str, SourceConfig],
    credentials: Mapping[str, Credential] | None = None,
    base: DataAccessCollection | None = None,
) -> DataAccessCollection:
    """A copy of ``base`` (or a fresh collection) with the files and folders of ``sources`` and ``credentials``."""
    files = sorted({file for source in sources.values() for file in source.files})
    folders = sorted({folder for source in sources.values() for folder in source.folders})
    collection = DataAccessCollection() if base is None else base.copy()
    for file in files:
        collection.add_file(file)
    for folder in folders:
        collection.add_folder(folder)
    for handle, credential in (credentials or {}).items():
        collection.add_credentials(handle, credential)
    return collection
