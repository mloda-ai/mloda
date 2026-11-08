"""Divergence record: how a plugin's result differs from its reference contract, and Cause for switched-off tests."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum

MAX_REASON_LENGTH = 200
MAX_DOCS_LENGTH = 500
ANY_OUTPUT_PART = "*"  # mirrors ANY_OUTPUT_PART in feature_chain_parser


class Cause(Enum):
    """Why a divergence exists, or why a host redefines, blanks or skips an inherited contract test."""

    NOT_IMPLEMENTED = "not_implemented"
    BACKEND = "backend"
    ENVIRONMENT = "environment"
    OTHER = "other"


class DivergenceKind(Enum):
    VALUE = "value"
    TYPE = "type"
    VARIANT = "variant"
    GAP = "gap"


class DivergenceSource(Enum):
    FEATURE_GROUP = "feature_group"
    IMPLEMENTATION = "implementation"


def _single_line_problem(label: str, text: str) -> str | None:
    lines = text.splitlines()
    if text and (len(lines) != 1 or lines[0] != text):
        return f"{label} must be a single line"
    if not text.isprintable():
        return f"{label} must contain only printable characters"
    limit = MAX_DOCS_LENGTH if label == "docs" else MAX_REASON_LENGTH
    if len(text) > limit:
        return f"{label} must be at most {limit} characters"
    return None


def _value_problems(record: Divergence) -> list[str]:
    problems: list[str] = []
    reason, variant, docs = record.reason, record.variant, record.docs
    if (
        not isinstance(reason, str)
        or (reason and not reason.strip())
        or (not reason and record.cause is not Cause.NOT_IMPLEMENTED)
    ):
        problems.append("gives no reason")
    for label, text in (("reason", reason), ("variant", variant), ("docs", docs)):
        if isinstance(text, str):
            problem = _single_line_problem(label, text)
            if problem:
                problems.append(problem)
    if record.cause is Cause.OTHER and not (isinstance(docs, str) and docs):
        problems.append("cause OTHER needs non-empty docs")
    if record.kind is DivergenceKind.VARIANT and not (isinstance(variant, str) and variant):
        problems.append("kind VARIANT needs a non-empty variant")
    if record.kind is not DivergenceKind.VARIANT and variant is not None:
        problems.append("variant is only allowed for kind VARIANT")
    if record.subtypes and record.outputs:
        problems.append("declare subtypes or outputs, not both")
    return problems


def _names(label: str, value: object) -> frozenset[str]:
    if isinstance(value, str | Mapping) or not hasattr(value, "__iter__"):
        raise TypeError(f"{label} must be an iterable of strings, got {value!r}")
    names = frozenset(value)
    if not all(isinstance(n, str) for n in names):
        raise TypeError(f"{label} must contain only strings, got {value!r}")
    if not all(names):
        raise ValueError(f"{label} must not contain an empty string")
    return names


@dataclass(frozen=True, kw_only=True)
class Divergence:
    """One declared difference from the reference contract, for a whole plugin, some subtypes or some outputs."""

    kind: DivergenceKind
    cause: Cause
    subtypes: frozenset[str] = frozenset()
    outputs: frozenset[str] = frozenset()
    reason: str = ""
    variant: str | None = None
    docs: str | None = None
    source: DivergenceSource | None = field(init=False, default=None)
    framework: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DivergenceKind):
            raise TypeError(f"kind must be a DivergenceKind, got {self.kind!r}")
        if not isinstance(self.cause, Cause):
            raise TypeError(f"cause must be a Cause, got {self.cause!r}")
        object.__setattr__(self, "subtypes", _names("subtypes", self.subtypes))
        object.__setattr__(self, "outputs", _names("outputs", self.outputs))
        for label, value in (("reason", self.reason), ("variant", self.variant), ("docs", self.docs)):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{label} must be a string, got {value!r}")
        problems = _value_problems(self)
        if problems:
            raise ValueError("; ".join(problems))

    def describe(self) -> str:
        """The reason, or "not implemented" for a reasonless NOT_IMPLEMENTED record."""
        return self.reason or "not implemented"

    def applies_to(self, subtype: str | None, parts: frozenset[str] | None) -> bool:
        """Whether the record covers a feature with this canonical subtype and requested output parts."""
        if self.subtypes:
            return subtype in self.subtypes
        if self.outputs:
            return parts is None or bool(parts & self.outputs)
        return True


def stamped(record: Divergence, source: DivergenceSource, framework: str | None) -> Divergence:
    result = copy.copy(record)
    object.__setattr__(result, "source", source)
    object.__setattr__(result, "framework", framework)
    return result


def structure_problems(owner: str, records: object) -> list[str]:
    """Problems with the container and its entries being Divergence records."""
    if not isinstance(records, tuple):
        return [f"{owner} divergences must be a tuple of Divergence, got {records!r}"]
    return [
        f"{owner} divergences entry must be a Divergence, got {r!r}" for r in records if not isinstance(r, Divergence)
    ]


def needs_universe(records: object) -> bool:
    """Whether any record names subtypes, so the subtype universe must be computed."""
    return isinstance(records, tuple) and any(isinstance(r, Divergence) and r.subtypes for r in records)


def divergence_problems(
    owner: str, records: object, universe: frozenset[str], output_parts: tuple[str, ...] | None
) -> list[str]:
    """Every problem with a divergence declaration; never raises on arbitrary input."""
    problems = structure_problems(owner, records)
    if not isinstance(records, tuple):
        return problems
    for record in records:
        if not isinstance(record, Divergence):
            continue
        outside = sorted(record.subtypes - universe)
        if outside:
            problems.append(f"{owner} divergences name {outside} outside the subtype universe")
        if output_parts is not None and ANY_OUTPUT_PART not in output_parts:
            unknown = sorted(record.outputs - set(output_parts))
            if unknown:
                problems.append(f"{owner} divergences name {unknown} outside OUTPUT_PARTS")
        problems.extend(f"{owner} divergences record {p}" for p in _value_problems(record))
    return problems


def validate_divergences(
    owner: str,
    records: object,
    universe: Callable[[], frozenset[str]],
    output_parts: tuple[str, ...] | None,
) -> None:
    """Raise TypeError for container or entry problems, else ValueError; universe is computed only if needed."""
    shape = structure_problems(owner, records)
    if shape:
        raise TypeError("; ".join(shape))
    problems = divergence_problems(owner, records, universe() if needs_universe(records) else frozenset(), output_parts)
    if problems:
        raise ValueError("; ".join(problems))
