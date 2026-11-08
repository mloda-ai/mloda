"""The configured sources as mloda sees them: their files, readers, columns and data access collection."""

import inspect
import logging
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
_LOGGER = logging.getLogger(__name__)


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
        with open(ref.path, "rb") as stream:
            raw = stream.read()
        # Decoding errors carry the offending bytes, so they are replaced by a message naming only the file.
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ConfigError(f"{where}: file {ref.path} is not valid UTF-8") from None
        value = text.removesuffix("\n").removesuffix("\r")
        if not value.strip():
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


def tolerant_credential_values(
    credentials: Mapping[str, Mapping[str, CredentialRef]], env: Mapping[str, str]
) -> tuple[dict[str, Credential], dict[str, str]]:
    """Read every credential; a handle with an unreadable field is left out and described in the problems."""
    values: dict[str, Credential] = {}
    problems: dict[str, str] = {}
    for handle, fields in credentials.items():
        # A ConfigError names the handle and field, never a value.
        try:
            values[handle] = Credential(
                {name: _credential_value(handle, name, ref, env) for name, ref in fields.items()}
            )
        except ConfigError as error:
            problems[handle] = str(error)
    return values, problems


class ServiceCredentials:
    """The service's credentials, re-read per call; touched only under MLODA_LOCK."""

    def __init__(
        self,
        credentials: Mapping[str, Mapping[str, CredentialRef]],
        env: Mapping[str, str],
        base: DataAccessCollection | None = None,
    ) -> None:
        self._credentials = credentials
        self._env = env
        self._layer = DataAccessCollection() if base is None else base.copy()
        self._registered: dict[str, Credential] = {}
        self._unreadable: set[str] = set()

    def collection(self, sources: Mapping[str, SourceConfig]) -> tuple[DataAccessCollection, dict[str, str]]:
        """A fresh collection of ``sources`` and the readable credentials, with the problems of the others."""
        values, problems = tolerant_credential_values(self._credentials, self._env)
        self._log_transitions(problems)
        for handle in [handle for handle in self._registered if handle not in values]:
            del self._registered[handle]
            del self._layer.credentials[handle]
        for handle, credential in values.items():
            if self._registered.get(handle) != credential:
                self._layer = self._layer.with_credentials({handle: credential})
                self._registered[handle] = credential
        return data_access_collection(sources, None, self._layer), problems

    def _log_transitions(self, problems: Mapping[str, str]) -> None:
        for handle, problem in problems.items():
            if handle not in self._unreadable:
                _LOGGER.warning("Credential unreadable, requests that read it fail: %s", problem)
        for handle in self._unreadable - set(problems):
            _LOGGER.info("Credential '%s' is readable again.", handle)
        self._unreadable = set(problems)


def check_credentials(
    settings: Settings, env: Mapping[str, str], base: DataAccessCollection | None = None
) -> str | None:
    """The problem with the configured credentials (unresolvable, or a handle ``base`` holds), else None."""
    # credential_values signals an unresolvable credential only by raising ConfigError.
    try:
        credential_values(settings.credentials, env)
    except ConfigError as error:
        return str(error)
    taken = sorted(set(settings.credentials) & set(base.handles() if base is not None else ()))
    if taken:
        return f"[mcp.credentials.{taken[0]}]: the handle is already registered in the given collection"
    return None


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
