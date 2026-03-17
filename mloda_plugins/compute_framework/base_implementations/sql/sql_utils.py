from typing import Any, Tuple


def quote_ident(name: str) -> str:
    """Quote a SQL identifier, escaping embedded double quotes."""
    return f'"{name.replace(chr(34), chr(34) + chr(34))}"'


def quote_value(value: Any) -> str:
    """Quote a SQL literal value safely. Fallback for when parameterized queries are unavailable."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def inline_params(condition: str, params: Tuple[Any, ...]) -> str:
    """Replace ? placeholders with quoted values. Fallback when no connection is available."""
    result = condition
    for p in params:
        result = result.replace("?", quote_value(p), 1)
    return result
