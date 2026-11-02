"""Credential handle option spec and its plan-time validation.

A declared handle option names a credential registered in the DataAccessCollection. Core checks the
handle at plan time; nothing reads the credential at run time yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mloda.core.abstract_plugins.components.property_spec import NO_DEFAULT, PropertySpec
from mloda.core.abstract_plugins.components.utils import escalate_match_abort

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
    from mloda.core.abstract_plugins.components.options import Options
    from mloda.core.abstract_plugins.feature_group import FeatureGroup


def _handle_element_validator(value: Any) -> bool:
    """True for a non-empty str; any other value is a marked abort naming its type, never the value."""
    if isinstance(value, str) and value:
        return True
    error = ValueError(
        f"A credential handle must be a non-empty str naming a registered credential, got {type(value).__name__}."
    )
    error.__dict__["prefix_option_key"] = True
    raise escalate_match_abort(error)


def _field_names(spec: CredentialHandleSpec, name: str, value: Any) -> tuple[str, ...]:
    prefix = f"CredentialHandleSpec({spec.explanation!r})"
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ValueError(f"{prefix}: {name} must be a tuple of field names, got {type(value).__name__}.")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{prefix}: {name} must hold non-empty str field names.")
    return tuple(value)


@dataclass(frozen=True)
class CredentialHandleSpec(PropertySpec):
    """A PropertySpec whose value names a credential handle; the subclass type is the marker core reads."""

    required_fields: tuple[str, ...] = ()
    public_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        prefix = f"CredentialHandleSpec({self.explanation!r})"
        if self.context is not False or self.strict_validation is not True or self.caller_settable is not False:
            raise ValueError(f"{prefix}: context=False, strict_validation=True and caller_settable=False are fixed.")
        if self.element_validator is not _handle_element_validator:
            raise ValueError(f"{prefix}: the handle element validator is fixed.")
        object.__setattr__(self, "required_fields", _field_names(self, "required_fields", self.required_fields))
        object.__setattr__(self, "public_fields", _field_names(self, "public_fields", self.public_fields))


def credential_handle_spec(
    explanation: str,
    *,
    default: Any = NO_DEFAULT,
    required_fields: tuple[str, ...] = (),
    public_fields: tuple[str, ...] = (),
) -> CredentialHandleSpec:
    """Build a spec for an option naming a credential handle registered in the DataAccessCollection."""
    return CredentialHandleSpec(
        explanation,
        default=default,
        context=False,
        strict_validation=True,
        element_validator=_handle_element_validator,
        caller_settable=False,
        required_fields=required_fields,
        public_fields=public_fields,
    )


def credential_handle_keys(feature_group: type[FeatureGroup]) -> dict[str, CredentialHandleSpec]:
    """The group's declared credential handle option keys."""
    mapping = getattr(feature_group, "PROPERTY_MAPPING", None) or {}
    return {key: spec for key, spec in mapping.items() if isinstance(spec, CredentialHandleSpec)}


def _declared(feature_group: type[FeatureGroup], options: Options) -> dict[str, Any]:
    keys = credential_handle_keys(feature_group)
    if not keys:
        return {}
    resolved = feature_group.options_with_defaults(options)
    return {key: resolved.get(key) for key in keys}


def declared_credential_handles(feature_group: type[FeatureGroup], options: Options) -> tuple[str, ...]:
    """Sorted handle names a step declares; an unset optional handle declares nothing."""
    return tuple(sorted({v for v in _declared(feature_group, options).values() if isinstance(v, str)}))


def _is_safe_credential_key(key: str) -> bool:
    """Whether a bytes value under this key is allowed; one place for a later safe-key list."""
    return False


def _bytes_path(value: Any, path: str, seen: set[int]) -> str | None:
    if isinstance(value, (bytes, bytearray)):
        return path
    if id(value) in seen:
        return None
    children: list[tuple[str, Any]]
    if isinstance(value, Mapping):
        children = [(f"{path}.{k}" if path else str(k), v) for k, v in value.items()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = [(f"{path}[{i}]", v) for i, v in enumerate(value)]
    else:
        return None
    seen.add(id(value))
    for child_path, child in children:
        found = _bytes_path(child, child_path, seen)
        if found is not None:
            return found
    return None


def _refuse(message: str) -> ValueError:
    return escalate_match_abort(ValueError(message))


def _check_one(
    feature_group: type[FeatureGroup],
    key: str,
    spec: CredentialHandleSpec,
    handle: Any,
    dac: DataAccessCollection | None,
) -> None:
    group_name = feature_group.__name__
    if not isinstance(handle, str) or not handle:
        raise _refuse(
            f"Option '{key}' of {group_name} must be a non-empty str naming a credential handle, "
            f"got {type(handle).__name__}."
        )
    credentials: dict[str, Any] = {} if dac is None else dac.credentials
    auto = frozenset() if dac is None else dac.auto_handles
    if handle in auto:
        raise _refuse(
            f"Option '{key}' of {group_name} names an unnamed credential. Register the credential under a name "
            f"in the DataAccessCollection and name it here."
        )
    if handle not in credentials:
        actual_kind = None if dac is None else dac.handles().get(handle)
        if actual_kind is not None:
            raise _refuse(
                f"Option '{key}' of {group_name} names handle '{handle}', which is registered under kind "
                f"'{actual_kind}', but kind 'credentials' is required."
            )
        named = sorted(h for h in credentials if h not in auto)
        unnamed = len(credentials) - len(named)
        suffix = f" and {unnamed} unnamed" if unnamed else ""
        raise _refuse(
            f"Option '{key}' of {group_name} names handle '{handle}', which is not a registered credential. "
            f"Available credential handles: {named}{suffix}."
        )
    entry = credentials[handle]
    if spec.required_fields:
        present = entry if isinstance(entry, Mapping) else {}
        missing = [field for field in spec.required_fields if field not in present]
        if missing:
            raise _refuse(
                f"Credential handle '{handle}' for option '{key}' of {group_name} is missing required "
                f"fields: {missing}."
            )
    path = _bytes_path(entry, "", set())
    if path is not None and not _is_safe_credential_key(path):
        raise _refuse(
            f"Credential handle '{handle}' for option '{key}' of {group_name} holds bytes at '{path}'. "
            f"Store a hex or base64 str and decode it in the plugin."
        )


def validate_credential_handles(
    feature_group: type[FeatureGroup], options: Options, dac: DataAccessCollection | None
) -> None:
    """Refuse a bad declared credential handle with a marked abort; messages never hold a credential value."""
    keys = credential_handle_keys(feature_group)
    if not keys:
        return
    for key in keys:
        if key in options.context:
            raise _refuse(
                f"Option '{key}' declares a credential handle for {feature_group.__name__} and must be a "
                f"group option, not a context option."
            )
    for key, handle in _declared(feature_group, options).items():
        if handle is None:
            continue
        _check_one(feature_group, key, keys[key], handle, dac)
