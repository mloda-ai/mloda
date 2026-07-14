"""The PROPERTY_MAPPING spec type.

``PropertySpec`` is the typed, frozen spec and IS the contract: construction enforces every
invariant (issue #694). ``property_spec`` is a thin builder kept for its authoring surface;
its ``strict=`` keyword maps to the ``strict_validation=`` field.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


class _NoDefault:
    """Type of the ``NO_DEFAULT`` sentinel."""

    def __repr__(self) -> str:
        return "NO_DEFAULT"

    def __reduce__(self) -> str:
        # Pickle and deepcopy resolve back to the module-level singleton instead of cloning it.
        return "NO_DEFAULT"


NO_DEFAULT: Any = _NoDefault()
"""The spec declares NO default, which makes the key required.

``default=None`` is a DECLARED default of ``None``: it makes the key optional and applies no
value. The retired dict form drew that line with a PRESENT ``default: None`` entry; on a
dataclass every field is always present, so the sentinel carries the distinction instead.
"""


def is_no_default(value: Any) -> bool:
    """Is this the "declares no default" sentinel?

    A type test, not an identity test: a second imported copy of this module (editable install
    plus site-packages, importlib.reload) has its own sentinel object, and an identity test would
    read that copy's sentinel as a declared default.
    """
    return isinstance(value, _NoDefault)


# The declared type lists the container shapes so a bare str/bytes (the forgotten-comma bug) is
# an author-time type error rather than a silent substring value space. The runtime is
# deliberately more lenient: __post_init__ materializes ANY non-Mapping iterable (a generator,
# a range) to a tuple, and keeps the str/bytes ValueError as the backstop for unchecked callers.
# A Mapping is the value-space-with-descriptions form and is kept as given.
AllowedValues = Mapping[Any, str] | tuple[Any, ...] | list[Any] | set[Any] | frozenset[Any]


@dataclass(frozen=True)
class PropertySpec:
    """Typed, frozen PROPERTY_MAPPING spec. Construction enforces the spec invariants."""

    explanation: str
    allowed_values: AllowedValues | None = None
    default: Any = NO_DEFAULT
    context: bool = True
    strict_validation: bool = False
    element_validator: Callable[[Any], Any] | None = None
    match_guard: Callable[[Any], Any] | None = None
    required_when: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        prefix = f"PropertySpec({self.explanation!r})"

        # A str/bytes is iterable, so tuple("add") would silently build ('a', 'd', 'd'): reject the
        # shape before materializing it.
        if isinstance(self.allowed_values, (str, bytes)):
            raise ValueError(
                f"{prefix}: allowed_values is a {type(self.allowed_values).__name__} "
                f"({self.allowed_values!r}), which would make membership a substring test. Wrap it in a "
                f"container, for example a one-element tuple: ({self.allowed_values!r},)."
            )

        if self.allowed_values is not None and not isinstance(self.allowed_values, Mapping):
            if not isinstance(self.allowed_values, Iterable):
                raise ValueError(
                    f"{prefix}: allowed_values is a {type(self.allowed_values).__name__} "
                    f"({self.allowed_values!r}), not a Mapping or an iterable of accepted values."
                )
            object.__setattr__(self, "allowed_values", tuple(self.allowed_values))

        if not isinstance(self.strict_validation, bool):
            raise ValueError(
                f"{prefix}: strict_validation must be a bool, got {type(self.strict_validation).__name__}."
            )

        if self.element_validator is not None and not callable(self.element_validator):
            raise ValueError(f"{prefix}: element_validator must be callable.")

        if self.required_when is not None and not callable(self.required_when):
            raise ValueError(f"{prefix}: required_when must be callable.")

        if self.match_guard is not None and not callable(self.match_guard):
            raise ValueError(f"{prefix}: match_guard must be callable.")

        if self.element_validator is not None and not self.strict_validation:
            raise ValueError(f"{prefix}: element_validator is never enforced without strict_validation=True.")

        self._check_value_space(prefix)
        self._check_declared_default(prefix)

    def _check_value_space(self, prefix: str) -> None:
        """A strict spec needs something to validate AGAINST.

        An ``element_validator`` IS that value space: the validator, not membership, decides, so
        ``allowed_values`` is not consulted at all and its emptiness cannot reject anything.
        """
        if not self.strict_validation or self.element_validator is not None:
            return

        if self.allowed_values is not None and not self.allowed_values:
            raise ValueError(f"{prefix}: an empty allowed_values would reject every value.")

        if self.allowed_values is None:
            raise ValueError(
                f"{prefix}: strict_validation=True needs a non-empty allowed_values or an element_validator."
            )

    def _check_declared_default(self, prefix: str) -> None:
        """A strict spec's declared default must be within its own value space (issue #530 semantics).

        ``NO_DEFAULT`` declares no default, and a declared ``None`` applies no value: neither has
        anything to check.
        """
        if is_no_default(self.default) or self.default is None or not self.strict_validation:
            return

        if self.element_validator is not None:
            try:
                verdict = self.element_validator(self.default)
            except Exception as exc:
                raise ValueError(
                    f"{prefix}: the element_validator raised an error when called with default {self.default!r}."
                ) from exc
            if not verdict:
                raise ValueError(f"{prefix}: default {self.default!r} is rejected by the element_validator.")
            return

        if self.allowed_values is None:
            return

        # Mirrors the match path: an unhashable default can never be a member of a hash-based value
        # space, so the TypeError is a clean rejection, not an escaping error.
        try:
            is_member = self.default in self.allowed_values
        except TypeError:
            is_member = False
        if not is_member:
            raise ValueError(f"{prefix}: default {self.default!r} is not within the declared allowed_values.")


def property_spec(
    explanation: str,
    *,
    strict: bool = False,
    allowed_values: AllowedValues | None = None,
    default: Any = NO_DEFAULT,
    context: bool = True,
    element_validator: Callable[[Any], Any] | None = None,
    required_when: Callable[[Any], Any] | None = None,
    match_guard: Callable[[Any], Any] | None = None,
) -> PropertySpec:
    """Build a ``PropertySpec``; the ``strict=`` keyword sets the ``strict_validation`` field."""
    return PropertySpec(
        explanation,
        allowed_values=allowed_values,
        default=default,
        context=context,
        strict_validation=strict,
        element_validator=element_validator,
        match_guard=match_guard,
        required_when=required_when,
    )
