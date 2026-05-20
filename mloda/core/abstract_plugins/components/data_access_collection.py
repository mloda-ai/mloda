from typing import Any, Callable, Optional


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
        connections: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, str]] = None,
        folders: Optional[dict[str, str]] = None,
        credentials: Optional[dict[str, Any]] = None,
        column_to_file: Optional[dict[str, str]] = None,
    ) -> None:
        self.connections: dict[str, Any] = dict(connections) if connections is not None else {}
        self.files: dict[str, str] = dict(files) if files is not None else {}
        self.folders: dict[str, str] = dict(folders) if folders is not None else {}
        self.credentials: dict[str, Any] = dict(credentials) if credentials is not None else {}

        self._check_cross_kind_handle_uniqueness()

        if column_to_file is not None:
            for col, handle in column_to_file.items():
                if handle not in self.files:
                    raise ValueError(
                        f"column_to_file value '{handle}' for column '{col}' is not a file handle. "
                        f"Available file handles: {sorted(self.files.keys())}"
                    )
        self.column_to_file: Optional[dict[str, str]] = column_to_file

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

    def add_connection(self, handle: str, conn: Any) -> None:
        self._reject_duplicate(handle)
        self.connections[handle] = conn

    def add_file(self, handle: str, path: str) -> None:
        self._reject_duplicate(handle)
        self.files[handle] = path

    def add_folder(self, handle: str, path: str) -> None:
        self._reject_duplicate(handle)
        self.folders[handle] = path

    def add_credentials(self, handle: str, creds: Any) -> None:
        self._reject_duplicate(handle)
        self.credentials[handle] = creds

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
        predicate: Optional[Callable[[Any], bool]] = None,
        hint: Optional[str] = None,
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
        raise ValueError(
            f"Ambiguous resolve for kind '{kind}': {len(matches)} candidates {candidate_handles}; "
            f"set 'data_access_handle' in Options to disambiguate."
        )
