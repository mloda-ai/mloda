"""Reads a plugin's declared_attributes(features) into a validated, scalar-only, fresh dict."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from mloda.core.abstract_plugins.components.credential_scrub import scrub_credentials
from mloda.core.abstract_plugins.components.utils import safe_field_with_error


def read_declared_attributes(owner: Any, features: Any) -> dict[str, str | int | float | bool]:
    """Return owner.declared_attributes(features) as a fresh dict; raise TypeError on a non-Mapping or non-str key."""
    declared = owner.declared_attributes(features)
    if not isinstance(declared, Mapping):
        raise TypeError(f"declared_attributes must return a Mapping, got {type(declared).__name__}")
    result: dict[str, str | int | float | bool] = {}
    for key, value in declared.items():
        if not isinstance(key, str):
            raise TypeError(f"declared_attributes keys must be str, got {type(key).__name__}")
        if isinstance(value, bool):
            result[key] = bool(value)
        elif isinstance(value, int):
            result[key] = int(value)
        elif isinstance(value, float):
            result[key] = float(value)
        elif isinstance(value, str):
            result[key] = str.__str__(value)
    return result


def unmet_declaration_reason(
    consumer: str, owner: str, declared: Mapping[str, Any], required: Mapping[str, Any]
) -> str | None:
    """None when declared meets required; a None required value means any declared value (equal and same type else)."""
    for key, want in required.items():
        if key not in declared:
            have = "none" if not declared else "no such key"
            return f"{consumer} requires declared '{key}'; {owner} declares {have}"
        got = declared[key]
        if want is not None and not (got == want and type(got) is type(want)):
            return f"{consumer} requires declared '{key}' == {want!r}; {owner} declares {got!r}"
    return None


_Declarations = dict[type, tuple[dict[str, str | int | float | bool], str | None]]


def contained_declarations(owner: type, memo: _Declarations) -> tuple[dict[str, str | int | float | bool], str | None]:
    """Plan-time declarations of owner and the contained raise text, read once per memo."""
    if owner not in memo:
        fallback: dict[str, str | int | float | bool] = {}
        declared, error = safe_field_with_error(lambda: read_declared_attributes(owner, None), fallback)
        memo[owner] = (declared, None if error is None else scrub_credentials(error))
    return memo[owner]


@dataclass(frozen=True)
class DeclarationRequirement:
    """A consumer's requirements plus the candidate feature group's own declarations."""

    consumer: str
    required: Mapping[str, Any]
    owner: type
    memo: _Declarations

    def unmet_reason(self, owner_name: str, reader: type | None) -> str | None:
        """Reason the merged {group, reader} declarations miss the requirement, else None; reads them lazily."""
        owner_declared, error = contained_declarations(self.owner, self.memo)
        declared = dict(owner_declared)
        if reader is not None:
            reader_declared, reader_error = contained_declarations(reader, self.memo)
            declared.update(reader_declared)
            error = error or reader_error
        if error is not None:
            return f"{self.consumer} requires declarations of {owner_name}; reading them raised: {error}"
        return unmet_declaration_reason(self.consumer, owner_name, declared, self.required)


_CURRENT: ContextVar[DeclarationRequirement | None] = ContextVar("mloda_declaration_requirement", default=None)


def current_declaration_requirement() -> DeclarationRequirement | None:
    return _CURRENT.get()


@contextmanager
def declaration_requirement_scope(requirement: DeclarationRequirement | None) -> Iterator[None]:
    """Make requirement (or none) current for reader selection inside the block."""
    token = _CURRENT.set(requirement)
    try:
        yield
    finally:
        _CURRENT.reset(token)
