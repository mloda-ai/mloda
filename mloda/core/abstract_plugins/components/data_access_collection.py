from typing import Any, Callable

from mloda.core.abstract_plugins.components.credential import Credential
from mloda.core.abstract_plugins.components.hashable_dict import HashableDict


_KIND_TO_ATTR: dict[str, str] = {
    "connection": "connections",
    "file": "files",
    "folder": "folders",
    "credentials": "credentials",
}


class DataAccessCollection:
    """Registry of data resources keyed by stable handle names.

    Consumers bind one resource via ``resolve(...)``, which raises on ambiguity
    rather than letting non-deterministic iteration decide.

    See ``docs/docs/in_depth/named-data-access-handles.md``.
    """

    def __init__(
        self,
        connections: dict[str, Any] | set[Any] | list[Any] | None = None,
        files: dict[str, str] | set[str] | list[str] | None = None,
        folders: dict[str, str] | set[str] | list[str] | None = None,
        credentials: dict[str, Any] | list[Any] | Credential | None = None,
        column_to_file: dict[str, str] | None = None,
    ) -> None:
        """Build the collection from named or auto-named resources.

        Accepted shapes, by intent:
          * ``dict[handle, value]``: named resources, for multi-source setups
            where a consumer is pointed at one entry via
            ``Options(data_access_handle=...)``.
          * ``set`` / ``list`` of bare values: single-source convenience;
            entries get internal auto-handles the user never references.
          * ``credentials`` additionally accepts a single ``Credential`` (or
            ``Credential`` entries in the list/dict forms). A credential is
            itself a dict, so the bare ``{connector_id: slot}`` shape would be
            read as ``{handle: value}``; the typed form removes that ambiguity
            and is unwrapped to a plain dict at registration. Named-form
            credential values must be mappings (``dict`` or ``Credential``);
            anything else raises an early mis-wrap ``ValueError``.
            ``HashableDict`` is no longer accepted on the credentials path; the
            class itself stays for hashability-required internals (e.g. ``Options``
            hashing, ``ApiInputDataCollection``). Credentials have no ``set``
            form (dicts are unhashable).

        ``column_to_file`` maps a column to either a file handle or a file path
        (paths are normalized to their handle).
        """
        self._auto_handles: set[str] = set()

        credentials = self._normalize_credentials(credentials)

        # Start with user-supplied dict entries; collect non-dict inputs for a second pass
        # so that auto-handle numbering can dodge every user-supplied handle (across all kinds).
        self.connections: dict[str, Any] = self._copy_if_dict(connections)
        self.files: dict[str, str] = self._copy_if_dict(files)
        self.folders: dict[str, str] = self._copy_if_dict(folders)
        self.credentials: dict[str, Any] = self._copy_if_dict(credentials)

        self._check_cross_kind_handle_uniqueness()

        self._assign_auto_handles("connection", self.connections, connections)
        self._assign_auto_handles("file", self.files, files)
        self._assign_auto_handles("folder", self.folders, folders)
        self._assign_auto_handles("credentials", self.credentials, credentials)

        self.column_to_file: dict[str, str] | None = self._normalize_column_to_file(column_to_file)

    @classmethod
    def _normalize_credentials(
        cls, credentials: dict[str, Any] | list[Any] | Credential | None
    ) -> dict[str, Any] | list[Any] | None:
        if credentials is None:
            return None
        if isinstance(credentials, Credential):
            return [credentials.data]
        if isinstance(credentials, dict):
            context_keys = tuple(credentials.keys())
            return {
                handle: cls._validated_credential_value(handle, value, context_keys)
                for handle, value in credentials.items()
            }
        return [cls._unwrap_credential(entry) for entry in credentials]

    @staticmethod
    def _unwrap_credential(value: Any) -> Any:
        if isinstance(value, Credential):
            value = value.data
        if isinstance(value, HashableDict):
            raise ValueError(
                "HashableDict is no longer accepted as a credential value. "
                "Pass Credential({...}) or a plain dict instead. "
                "(HashableDict itself is not removed; it stays for hashability-required "
                "internals such as Options hashing.)"
            )
        return value

    @classmethod
    def _validated_credential_value(cls, handle: str, value: Any, context_keys: tuple[str, ...] | None = None) -> Any:
        unwrapped = cls._unwrap_credential(value)
        if isinstance(unwrapped, dict):
            return unwrapped
        fields = ", ".join(f"'{key}': ..." for key in (context_keys if context_keys is not None else (handle,)))
        raise ValueError(
            f"credentials value for handle '{handle}' is not a mapping. A bare dict is read as "
            f"{{handle: credential}}, so a single credential must be wrapped: use the list form "
            f"credentials=[{{{fields}}}], the typed form credentials=Credential({{{fields}}}), "
            f"or the named form credentials={{'my-db': {{{fields}}}}}."
        )

    @staticmethod
    def _copy_if_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _assign_auto_handles(self, kind: str, registry: dict[str, Any], raw: Any) -> None:
        if raw is None or isinstance(raw, dict):
            return
        for entry in raw:
            handle = self._next_auto_handle(kind)
            registry[handle] = entry
            self._auto_handles.add(handle)

    def _next_auto_handle(self, kind: str) -> str:
        counter = 0
        handle = f"_auto_{kind}_{counter}"
        while self._handle_taken(handle):
            counter += 1
            handle = f"_auto_{kind}_{counter}"
        return handle

    def _handle_taken(self, handle: str) -> bool:
        for attr in _KIND_TO_ATTR.values():
            registry: dict[str, Any] = getattr(self, attr)
            if handle in registry:
                return True
        return False

    def _normalize_column_to_file(self, column_to_file: dict[str, str] | None) -> dict[str, str] | None:
        if column_to_file is None:
            return None
        path_to_handle: dict[str, str] = {}
        for handle, path in self.files.items():
            if path not in path_to_handle:
                path_to_handle[path] = handle
        normalized: dict[str, str] = {}
        for col, value in column_to_file.items():
            if value in self.files:
                normalized[col] = value
            elif value in path_to_handle:
                normalized[col] = path_to_handle[value]
            else:
                raise ValueError(
                    f"column_to_file value '{value}' for column '{col}' is not a known file handle or path. "
                    f"Available file handles: {sorted(self.files.keys())}. "
                    f"Available file paths: {sorted(self.files.values())}."
                )
        return normalized

    def _check_cross_kind_handle_uniqueness(self) -> None:
        seen: dict[str, str] = {}
        for kind, attr in _KIND_TO_ATTR.items():
            registry: dict[str, Any] = getattr(self, attr)
            for handle in registry:
                if handle in seen:
                    raise ValueError(
                        f"Handle '{handle}' is registered under both kind '{seen[handle]}' and kind '{kind}'. "
                        f"Handles must be globally unique across kinds."
                    )
                seen[handle] = kind

    def add_connection(self, *args: Any) -> None:
        """Register a connection. ``add_connection(value)`` auto-names it; ``add_connection(handle, value)`` names it."""
        handle, value = self._resolve_mutator_args("connection", args)
        self._reject_duplicate(handle)
        self.connections[handle] = value

    def add_file(self, *args: Any) -> None:
        """Register a file. ``add_file(path)`` auto-names it; ``add_file(handle, path)`` names it."""
        handle, value = self._resolve_mutator_args("file", args)
        self._reject_duplicate(handle)
        self.files[handle] = value

    def add_folder(self, *args: Any) -> None:
        """Register a folder. ``add_folder(path)`` auto-names it; ``add_folder(handle, path)`` names it."""
        handle, value = self._resolve_mutator_args("folder", args)
        self._reject_duplicate(handle)
        self.folders[handle] = value

    def add_credentials(self, *args: Any) -> None:
        """Register credentials. ``add_credentials(value)`` auto-names them; ``add_credentials(handle, value)`` names them."""
        handle, value = self._resolve_mutator_args("credentials", args)
        self._reject_duplicate(handle)
        self.credentials[handle] = self._validated_credential_value(handle, value)

    def _resolve_mutator_args(self, kind: str, args: tuple[Any, ...]) -> tuple[str, Any]:
        if len(args) == 1:
            handle = self._next_auto_handle(kind)
            self._auto_handles.add(handle)
            return handle, args[0]
        if len(args) == 2:
            return args[0], args[1]
        raise TypeError(f"add_{kind}() takes 1 (value) or 2 (handle, value) positional arguments, got {len(args)}.")

    def _reject_duplicate(self, handle: str) -> None:
        for kind, attr in _KIND_TO_ATTR.items():
            if handle in getattr(self, attr):
                raise ValueError(
                    f"Handle '{handle}' is already registered under kind '{kind}'. Handles must be unique across kinds."
                )

    def handles(self) -> dict[str, str]:
        """Return ``{handle: kind}`` for every registered resource."""
        result: dict[str, str] = {}
        for kind, attr in _KIND_TO_ATTR.items():
            for handle in getattr(self, attr):
                result[handle] = kind
        return result

    def resolve(
        self,
        kind: str,
        predicate: Callable[[Any], bool] | None = None,
        hint: str | None = None,
    ) -> Any:
        """Resolve a single resource of ``kind``, optionally narrowed by ``predicate``.

        Hint semantics:
          * missing handle: ValueError naming kind + handle + available handles of that kind
          * wrong kind: ValueError naming actual vs requested kind
          * predicate fails: ValueError naming predicate mismatch
          * otherwise: return the entry

        No-hint semantics: 0 matches -> None, 1 match -> the entry,
        >1 matches -> ValueError listing candidate handles and mentioning ``data_access_handle``.
        """
        attr = _KIND_TO_ATTR.get(kind)
        if attr is None:
            raise ValueError(f"Unknown kind '{kind}'. Expected one of {sorted(_KIND_TO_ATTR.keys())}.")
        registry: dict[str, Any] = getattr(self, attr)

        if hint is not None:
            if hint not in registry:
                actual_kind = self.handles().get(hint)
                if actual_kind is not None:
                    raise ValueError(
                        f"Handle '{hint}' is registered under kind '{actual_kind}', but kind '{kind}' was requested."
                    )
                raise ValueError(
                    f"Handle '{hint}' not found for kind '{kind}'. "
                    f"Available handles of kind '{kind}': {sorted(registry.keys())}"
                )
            entry = registry[hint]
            if predicate is not None and not predicate(entry):
                raise ValueError(
                    f"Handle '{hint}' of kind '{kind}' did not satisfy the predicate (predicate mismatch)."
                )
            return entry

        if predicate is None:
            matches: list[tuple[str, Any]] = list(registry.items())
        else:
            matches = [(h, v) for h, v in registry.items() if predicate(v)]

        if not matches:
            return None
        if len(matches) == 1:
            return matches[0][1]

        candidate_handles = [h for h, _ in matches]
        all_auto = all(h in self._auto_handles for h in candidate_handles)
        if all_auto:
            if kind == "credentials":
                bullets = "\n".join(f"  - {self._redacted_credential(v)}" for _, v in matches)
            else:
                bullets = "\n".join(f"  - {v}" for _, v in matches)
            raise ValueError(
                f"Ambiguous resolve for kind '{kind}': {len(matches)} matches:\n"
                f"{bullets}\n"
                f"Name them and set Options(data_access_handle=...) to pick one, "
                f"or remove one from the collection."
            )
        raise ValueError(
            f"Ambiguous resolve for kind '{kind}': {len(matches)} candidates {candidate_handles}; "
            f"set 'data_access_handle' in Options to disambiguate."
        )

    @staticmethod
    def _redacted_credential(value: Any) -> str:
        if isinstance(value, dict):
            return "{" + ", ".join(f"'{key}': '***'" for key in value) + "}"
        return "'***'"
