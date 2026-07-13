"""Authoring helper that builds a conventional PROPERTY_MAPPING spec dict.

The contract stays a plain dict; this only changes authoring, validating the
invariants up front instead of relying on hand-written spec dicts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser


def property_spec(
    explanation: str,
    *,
    strict: bool = False,
    allowed_values: Mapping[Any, str] | Iterable[Any] | None = None,
    default: Any = None,
    context: bool = True,
    element_validator: Callable[..., Any] | None = None,
    required_when: Callable[..., Any] | None = None,
    match_guard: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    # A str/bytes is iterable, so tuple("add") would silently build ('a', 'd', 'd'): reject the
    # shape before materializing it. Holds whether or not the spec is strict.
    if isinstance(allowed_values, (str, bytes)):
        raise ValueError(
            f"property_spec({explanation!r}): allowed_values is a {type(allowed_values).__name__} "
            f"({allowed_values!r}), which would make membership a substring test. Wrap it in a "
            f"container, for example a one-element tuple: ({allowed_values!r},)."
        )

    if allowed_values is not None and not isinstance(allowed_values, Mapping):
        allowed_values = tuple(allowed_values)

    if element_validator is not None and not callable(element_validator):
        raise ValueError(f"property_spec({explanation!r}): element_validator must be callable.")

    if required_when is not None and not callable(required_when):
        raise ValueError(f"property_spec({explanation!r}): required_when must be callable.")

    if match_guard is not None and not callable(match_guard):
        raise ValueError(f"property_spec({explanation!r}): match_guard must be callable.")

    # Dead without strict: _validate_property_value returns early on a non-strict spec. allowed_values
    # is NOT dead there, so it has no such rule: a non-strict value space still maps a name-parsed
    # value back onto its PROPERTY_MAPPING key.
    if element_validator is not None and not strict:
        raise ValueError(f"property_spec({explanation!r}): element_validator is never enforced without strict=True.")

    if strict and allowed_values is not None and not allowed_values:
        raise ValueError(f"property_spec({explanation!r}): an empty allowed_values would reject every value.")

    if strict and allowed_values is None and element_validator is None:
        raise ValueError(
            f"property_spec({explanation!r}): strict=True needs a non-empty allowed_values or an element_validator."
        )

    spec: dict[str, Any] = {"explanation": explanation}
    if allowed_values is not None:
        spec[DefaultOptionKeys.allowed_values] = allowed_values
    if element_validator is not None:
        spec[DefaultOptionKeys.element_validator] = element_validator
    if required_when is not None:
        spec[DefaultOptionKeys.required_when] = required_when
    if match_guard is not None:
        spec[DefaultOptionKeys.match_guard] = match_guard
    spec[DefaultOptionKeys.context] = context
    spec[DefaultOptionKeys.strict_validation] = strict
    if default is not None:
        spec[DefaultOptionKeys.default] = default

    # The declared-default semantics live in core, so the builder and the
    # class-definition check cannot drift apart.
    FeatureChainParser.check_declared_default(f"property_spec({explanation!r})", explanation, spec)

    return spec
