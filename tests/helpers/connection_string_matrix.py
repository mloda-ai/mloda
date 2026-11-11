"""Connection strings with a secret-named parameter across prefix, separator, spacing, value form and tail."""

import itertools
from collections.abc import Iterator

SECRET = "ZQXSECRET"  # nosec B105

PREFIXES = [
    "jdbc:mysql://h/db",
    "JDBC:MYSQL://h/db",
    "Jdbc:h2:tcp://h/db",
    "jdbc:sqlserver://h",
    "s3://b/x",
    "https://h/x",
]
PARAMETERS = [";", "?", "?a=1&", "#"]
KEYS = ["password", "pass%77ord", "auth"]
SPACINGS = ["=", " = ", " =", "= "]
VALUES = {
    "plain": SECRET,
    "single-quoted-space": f"'A {SECRET}'",
    "double-quoted-space": f'"A {SECRET}"',
    "braced-space": f"{{A {SECRET}}}",
    "doubled-single-quote": f"'A''B {SECRET}'",
    "doubled-double-quote": f'"A""B {SECRET}"',
    "brace-escape": f"{{A}}}}B {SECRET}}}",
    "bearer": f"Bearer {SECRET}",
}
TRAILINGS = {
    "end": ("", ""),
    "prose": ("", " and then more text"),
    "period": ("", "."),
    "json-list-item": ('["', '", "next"]'),
    "dict-repr": ("{'url': '", "'}"),
    "literal-newline": ("", "\\nhost=x"),
}
HOST_STOPS = ["!", "{", ",", "|", "*", "+", "$", "^", "("]

# Combinations left out of the product, each with the reason.
EXCLUSIONS = {
    "unquoted multi-word values": "indistinguishable from prose after the secret",
    "bearer value without a space after the separator": "the token then reads as a plain word, covered by plain",
    "quote-wrapped value in a dict repr or json item with a matching quote": "the wrapper quote ends the value early",
}


def _excluded(trailing: str, value: str, spacing: str) -> bool:
    quoted = value[0] in "'\""
    wrapper_quote = TRAILINGS[trailing][0][-1:]
    if quoted and wrapper_quote == value[0] and trailing in ("json-list-item", "dict-repr"):
        return True
    return value.startswith("Bearer") and spacing.endswith("=")


def connection_string_matrix() -> Iterator[tuple[str, str, str, str]]:
    """Yield (label, text, expected_unused, secret)."""
    for prefix, key, value_name, trailing in itertools.product(PREFIXES, KEYS, VALUES, TRAILINGS):
        for parameter, spacing in itertools.product(PARAMETERS, SPACINGS):
            if _excluded(trailing, VALUES[value_name], spacing):
                continue
            lead, tail = TRAILINGS[trailing]
            text = f"{lead}{prefix}{parameter}{key}{spacing}{VALUES[value_name]}{tail}"
            label = (
                f"prefix={prefix} param={parameter!r} key={key} spacing={spacing!r} value={value_name} tail={trailing}"
            )
            yield label, text, "", SECRET
    for stop, trailing in itertools.product(HOST_STOPS, TRAILINGS):
        lead, tail = TRAILINGS[trailing]
        yield f"host-stop={stop!r} tail={trailing}", f"{lead}s3://b{stop}y;v={SECRET}{tail}", "", SECRET
