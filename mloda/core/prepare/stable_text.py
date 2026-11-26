"""Text of a value that is equal across interpreters."""

import re

from mloda.core.abstract_plugins.components.options import Options, _str_option_dict
from mloda.core.abstract_plugins.components.plan_freeze import ReadOnlyDict


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def stable_text(value: object) -> str:
    """Text with addresses removed, set elements sorted and a container nested in itself written as repr does."""
    return _stable_text(value, frozenset())


def _stable_text(value: object, path: frozenset[int]) -> str:
    if isinstance(value, Options):
        if id(value) in path:
            return "Options(...)"
        inner = path | {id(value)}
        parts = (
            f"group={_stable_text(_str_option_dict(value.group), inner)}, "
            f"context={_stable_text(_str_option_dict(value.context), inner)}"
        )
        if value.propagate_context_keys:
            parts += f", propagate_context_keys={_stable_text(value.propagate_context_keys, inner)}"
        if value.cascade_context_keys:
            parts += f", cascade_context_keys={_stable_text(value.cascade_context_keys, inner)}"
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
