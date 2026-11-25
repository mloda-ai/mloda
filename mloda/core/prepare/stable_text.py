"""Text of a value that is equal across interpreters."""

import re

from mloda.core.abstract_plugins.components.options import Options, _str_option_dict
from mloda.core.abstract_plugins.components.plan_freeze import ReadOnlyDict


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def stable_text(value: object) -> str:
    """Text of a value with addresses removed and set elements sorted, so it is equal across interpreters.

    A container met again inside itself is written as `[...]`, `{...}` or `(...)`, as `repr` does, so a
    self-referential option value has a text instead of raising `RecursionError`.
    """
    return _stable_text(value, frozenset())


def _stable_text(value: object, path: frozenset[int]) -> str:
    if isinstance(value, Options):
        parts = (
            f"group={_stable_text(_str_option_dict(value.group), path)}, "
            f"context={_stable_text(_str_option_dict(value.context), path)}"
        )
        if value.propagate_context_keys:
            parts += f", propagate_context_keys={_stable_text(value.propagate_context_keys, path)}"
        if value.cascade_context_keys:
            parts += f", cascade_context_keys={_stable_text(value.cascade_context_keys, path)}"
        return f"Options({parts})"
    if type(value) is dict or type(value) is ReadOnlyDict:
        if id(value) in path:
            return "{...}"
        inner = path | {id(value)}
        items = sorted((_stable_text(k, inner), _stable_text(v, inner)) for k, v in value.items())
        return "{" + ", ".join(f"{k}: {v}" for k, v in items) + "}"
    if type(value) is list:
        if id(value) in path:
            return "[...]"
        inner = path | {id(value)}
        return "[" + ", ".join(_stable_text(v, inner) for v in value) + "]"
    if type(value) is tuple:
        if id(value) in path:
            return "(...)"
        inner = path | {id(value)}
        return "(" + ", ".join(_stable_text(v, inner) for v in value) + ("," if len(value) == 1 else "") + ")"
    if type(value) is set or type(value) is frozenset:
        return "{" + ", ".join(sorted(_stable_text(v, path) for v in value)) + "}"
    return _ADDRESS.sub("", repr(value))
