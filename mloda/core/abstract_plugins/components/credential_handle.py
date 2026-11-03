"""Credential handle option spec and its plan-time validation.

A declared handle option names a credential registered in the DataAccessCollection. Core checks the
handle at plan time; nothing reads the credential at run time yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any

from mloda.core.abstract_plugins.components.credential_leaves import (
    credential_key_names,
    iter_credential_leaves,
    normalize_credential_key,
    public_key_names,
)
from mloda.core.abstract_plugins.components.property_spec import NO_DEFAULT, PropertySpec
from mloda.core.abstract_plugins.components.utils import PREFIX_OPTION_KEY_FLAG, escalate_match_abort

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
    error.__dict__[PREFIX_OPTION_KEY_FLAG] = True
    raise escalate_match_abort(error)


def _field_names(spec: CredentialHandleSpec, name: str, value: Any) -> tuple[str, ...]:
    prefix = f"CredentialHandleSpec({spec.explanation!r})"
    if name == "public_fields":
        return public_key_names(f"{prefix}: {name}", value)
    return credential_key_names(f"{prefix}: {name}", value)


@dataclass(frozen=True)
class CredentialHandleSpec(PropertySpec):
    """A PropertySpec whose value names a credential handle; the subclass type is the marker core reads."""

    required_fields: tuple[str, ...] = ()
    public_fields: tuple[str, ...] = ()
    forward_to_inputs: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        prefix = f"CredentialHandleSpec({self.explanation!r})"
        if (
            self.context is not False
            or self.strict_validation is not True
            or self.caller_settable is not False
            or self.forward_to_inputs is not False
        ):
            raise ValueError(
                f"{prefix}: context=False, strict_validation=True, caller_settable=False and "
                f"forward_to_inputs=False are fixed."
            )
        if self.element_validator is not _handle_element_validator:
            raise ValueError(f"{prefix}: the handle element validator is fixed.")
        for name in ("match_guard", "allowed_values", "required_when", "name_converter"):
            if getattr(self, name) is not None:
                raise ValueError(f"{prefix}: {name} must stay None for a credential handle.")
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
        forward_to_inputs=False,
        required_fields=required_fields,
        public_fields=public_fields,
    )


def credential_handle_keys(feature_group: type[FeatureGroup]) -> dict[str, CredentialHandleSpec]:
    """The group's declared credential handle option keys."""
    mapping = getattr(feature_group, "PROPERTY_MAPPING", None) or {}
    return {key: spec for key, spec in mapping.items() if isinstance(spec, CredentialHandleSpec)}


def public_credential_keys() -> frozenset[str]:
    """Normalized keys every loaded feature group declares as public, by PUBLIC_CREDENTIAL_KEYS or public_fields."""
    # Function-local: avoids an import cycle through the feature group module.
    from mloda.core.abstract_plugins.feature_group import feature_group_subclasses

    keys: set[str] = set()
    for group in feature_group_subclasses():
        keys.update(
            normalize_credential_key(key) for key in credential_key_names(group.__name__, group.PUBLIC_CREDENTIAL_KEYS)
        )
        for spec in credential_handle_keys(group).values():
            keys.update(normalize_credential_key(key) for key in spec.public_fields)
    return frozenset(keys)


def _declared(feature_group: type[FeatureGroup], options: Options) -> dict[str, Any]:
    keys = credential_handle_keys(feature_group)
    if not keys:
        return {}
    resolved = feature_group.options_with_defaults(options)
    return {key: resolved.get(key) for key in keys}


def declared_credential_handles(feature_group: type[FeatureGroup], options: Options) -> tuple[str, ...]:
    """Sorted handle names a step declares; an unset optional handle declares nothing."""
    return tuple(sorted({v for v in _declared(feature_group, options).values() if isinstance(v, str)}))


def _bytes_path(entry: Any) -> str | None:
    for item in iter_credential_leaves(entry):
        if item.secret and isinstance(item.leaf, (bytes, bytearray)):
            return item.path
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
    path = _bytes_path(entry)
    if path is not None:
        raise _refuse(
            f"Credential handle '{handle}' for option '{key}' of {group_name} holds bytes at '{path}'. "
            f"Store a hex or base64 str and decode it in the plugin."
        )


def _credential_hit(value: Any, seen: set[int]) -> str | None:
    """The credential type name found in value, descending containers, or None."""
    from mloda.core.abstract_plugins.components.credential import Credential, RegisteredCredential

    if isinstance(value, (Credential, RegisteredCredential)):
        return type(value).__name__
    if not isinstance(value, (Mapping, list, tuple, set, frozenset)) or id(value) in seen:
        return None
    seen.add(id(value))
    children = [*value.keys(), *value.values()] if isinstance(value, Mapping) else list(value)
    for child in children:
        hit = _credential_hit(child, seen)
        if hit is not None:
            return hit
    return None


def refuse_credential_option_values(options: Options) -> None:
    """Refuse a Credential or RegisteredCredential in any non-pointer option value, naming key and type only."""
    # Function-local: avoids an import cycle through the source feature group module.
    from mloda.core.abstract_plugins.components.input_data.source_feature_group import concrete_source_groups

    from mloda.core.abstract_plugins.components.credential import Credential, RegisteredCredential

    pointer_keys: Any = None
    for key, value in [*options.group.items(), *options.context.items()]:
        hit = _credential_hit(value, set())
        if hit is None:
            continue
        if pointer_keys is None:
            pointer_keys = concrete_source_groups().keys()
        if key in pointer_keys:
            continue
        direct = isinstance(value, (Credential, RegisteredCredential))
        where = "" if direct else " nested in its value"
        raise _refuse(
            f"Option '{key}' holds a {hit}{where}. A credential is never an option value: register it in the "
            f"DataAccessCollection and name its handle in an option declared with credential_handle_spec, "
            f"then read it with read_credential."
        )


def missing_credential_handle_abort(feature_group: type[FeatureGroup], options: Options) -> ValueError | None:
    """The marked abort for a required handle key that is absent, naming a close undeclared option key, or None."""
    # Function-local: avoids an import cycle through the feature chain parser.
    from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser

    keys = credential_handle_keys(feature_group)
    if not keys:
        return None
    mapping = feature_group.PROPERTY_MAPPING or {}
    absent = FeatureChainParser._name_path_missing_required_keys(feature_group.options_with_defaults(options), mapping)
    missing = [key for key in absent if key in keys]
    if not missing:
        return None
    candidates = [key for key in [*options.group, *options.context] if key not in mapping]
    sentences = []
    for key in missing:
        close = get_close_matches(key, candidates, n=1, cutoff=0.8)
        hint = f" (did you mean '{close[0]}'?)" if close else ""
        sentences.append(f"{feature_group.__name__} requires credential handle option '{key}', which is not set{hint}.")
    return _refuse(" ".join(sentences))


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
