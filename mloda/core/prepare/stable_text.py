"""Text of a value that is equal across interpreters."""

import re

from mloda.core.abstract_plugins.components.options import Options, _str_option_dict


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def stable_text(value: object) -> str:
    """Text of a value with addresses removed and set elements sorted, so it is equal across interpreters."""
    if isinstance(value, Options):
        parts = f"group={stable_text(_str_option_dict(value.group))}, context={stable_text(_str_option_dict(value.context))}"
        if value.propagate_context_keys:
            parts += f", propagate_context_keys={stable_text(value.propagate_context_keys)}"
        if value.cascade_context_keys:
            parts += f", cascade_context_keys={stable_text(value.cascade_context_keys)}"
        return f"Options({parts})"
    if type(value) is dict:
        items = sorted((stable_text(k), stable_text(v)) for k, v in value.items())
        return "{" + ", ".join(f"{k}: {v}" for k, v in items) + "}"
    if type(value) is list:
        return "[" + ", ".join(stable_text(v) for v in value) + "]"
    if type(value) is tuple:
        return "(" + ", ".join(stable_text(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    if type(value) is set or type(value) is frozenset:
        return "{" + ", ".join(sorted(stable_text(v) for v in value)) + "}"
    return _ADDRESS.sub("", repr(value))
