"""Authoring helper that builds a conventional PROPERTY_MAPPING spec dict.

The contract stays a plain dict; this only changes authoring, validating the
invariants up front instead of relying on hand-written spec dicts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys


def property_spec(
    explanation: str,
    *,
    strict: bool = False,
    allowed_values: Mapping[Any, str] | Iterable[Any] | None = None,
    default: Any = None,
    context: bool = True,
    validation_function: Callable[..., Any] | None = None,
    required_when: Callable[..., Any] | None = None,
    type_validator: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if allowed_values is not None and not isinstance(allowed_values, Mapping):
        allowed_values = tuple(allowed_values)

    if validation_function is not None and not callable(validation_function):
        raise ValueError(f"property_spec({explanation!r}): validation_function must be callable.")

    if required_when is not None and not callable(required_when):
        raise ValueError(f"property_spec({explanation!r}): required_when must be callable.")

    if type_validator is not None and not callable(type_validator):
        raise ValueError(f"property_spec({explanation!r}): type_validator must be callable.")

    if validation_function is not None and not strict:
        raise ValueError(f"property_spec({explanation!r}): validation_function is never enforced without strict=True.")

    if strict and allowed_values is not None and not allowed_values:
        raise ValueError(f"property_spec({explanation!r}): an empty allowed_values would reject every value.")

    if strict and allowed_values is None and validation_function is None:
        raise ValueError(
            f"property_spec({explanation!r}): strict=True needs a non-empty allowed_values or a validation_function."
        )

    if not strict and allowed_values is not None:
        raise ValueError(f"property_spec({explanation!r}): allowed_values is never enforced without strict=True.")

    if strict and default is not None:
        if validation_function is not None:
            try:
                verdict = validation_function(default)
            except Exception as exc:
                raise ValueError(
                    f"property_spec({explanation!r}): validation_function raised an error when called with default {default!r}."
                ) from exc
            if not verdict:
                raise ValueError(
                    f"property_spec({explanation!r}): default {default!r} is rejected by the validation_function."
                )
        elif allowed_values is not None:
            accepted = set(allowed_values.keys()) if isinstance(allowed_values, Mapping) else set(allowed_values)
            if default not in accepted:
                raise ValueError(
                    f"property_spec({explanation!r}): default {default!r} is not in the accepted set {accepted}."
                )

    spec: dict[str, Any] = {"explanation": explanation}
    if allowed_values is not None:
        spec[DefaultOptionKeys.allowed_values] = allowed_values
    if validation_function is not None:
        spec[DefaultOptionKeys.validation_function] = validation_function
    if required_when is not None:
        spec[DefaultOptionKeys.required_when] = required_when
    if type_validator is not None:
        spec[DefaultOptionKeys.type_validator] = type_validator
    spec[DefaultOptionKeys.context] = context
    spec[DefaultOptionKeys.strict_validation] = strict
    if default is not None:
        spec[DefaultOptionKeys.default] = default
    return spec
