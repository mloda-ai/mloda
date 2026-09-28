"""Strip credentials from free text before it reaches core's failure logs or MlodaRunError."""

import re
from collections.abc import Mapping
from typing import Any

_URI_PATTERN = re.compile(
    r"(?P<scheme>(?:jdbc:)?[A-Za-z][A-Za-z0-9+.-]*)://"
    r"(?:(?P<userinfo>[^/?#]*)@)?"
    r"(?P<host>(?:[\w.~-]*|\[[0-9A-Fa-f:.]+\])(?::\d+)?)"
    r"(?P<path>/[\w.~%/:=+-]*)?"
    r"(?:[?#][^@]*)?"
)
_AZURE_CONTAINER_SCHEMES = frozenset({"abfs", "abfss", "wasb", "wasbs"})
_AZURE_CONTAINER_PATTERN = re.compile(r"[a-z0-9-]{3,63}|\$(?:root|web|logs)")

# Lookbehind on the scheme, not a bare \b: without it, backtracking a scheme match that fails
# mid-alphanumeric-run against a long run of letters/digits is quadratic in the run's length.
_FREE_TEXT_URI_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+.-])(?P<scheme>(?:jdbc:)?[A-Za-z][A-Za-z0-9+.-]*)://"
    r"(?:(?P<userinfo>[^/?#\s]*)@)?"
    r"(?P<host>(?:\[[0-9A-Fa-f:.]+\]|[\w.~-]*)(?::\d+)?)"
    r"(?P<path>/[^\s?#\"'<>\]]*)?"
    r"(?:[?#][^\s\"'<>)\]]*)?"
)

# Same anti-backtracking lookbehind as above: only a scheme-shaped run right after a non-scheme
# char is even attempted.
_USERINFO_FALLBACK_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+.-])(?P<scheme>(?:jdbc:)?[A-Za-z][A-Za-z0-9+.-]*)://"
    r"[^\s/?#@:]*:(?!\d+(?:[/?#\s]|$))(?:(?!://)[^\s@])*@"
)

# (?<!\w), not (?<![A-Za-z]): a word-start anchor so a prefixed identifier (sslpassword,
# PGPASSWORD, DB_PASSWORD) still matches, captured whole so the prefix survives the replacement.
_PASSWORD_KEYWORD_PATTERN = re.compile(
    r"(?<!\w)(?P<keyword>\w*(?:password|passwd|pwd))(?P<sep>[ \t]*=[ \t]*)"
    r"(?:'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"|\{(?:[^}\n]|\}\})*\}|[^;&\s'\"]*)",
    re.IGNORECASE,
)


def _cut_key_value_path(path: str) -> str:
    """The path up to (excluding) the first segment that looks like key=value after a ':'."""
    segments = path.split("/")
    for index, segment in enumerate(segments):
        if "=" in segment.partition(":")[2]:
            return "/".join(segments[:index])
    return path


def _uri_projection(match: re.Match[str]) -> str | None:
    scheme, userinfo, host, path = match["scheme"], match["userinfo"], match["host"], match["path"] or ""
    container = ""
    if userinfo and scheme.lower() in _AZURE_CONTAINER_SCHEMES and _AZURE_CONTAINER_PATTERN.fullmatch(userinfo):
        container = f"{userinfo}@"
    if scheme.startswith("jdbc:"):
        return f"{scheme}://{container}{host}"
    if any("=" in segment.partition(":")[2] for segment in path.split("/")):
        return None
    head, percent, _ = path.partition("%")
    if percent:
        path = head[: head.rfind("/") + 1]
    return f"{scheme}://{container}{host}{path}"


def _scrub_uri_match(match: re.Match[str]) -> str:
    scheme, host, path = match["scheme"], match["host"], match["path"] or ""
    projection = _uri_projection(match)
    if projection is not None:
        return projection
    return f"{scheme}://{host}{_cut_key_value_path(path)}"


def _drop_free_text_userinfo(text: str) -> str:
    """Fallback for a userinfo-shaped `scheme://name:pwd@` prefix; drops through the first non-port `@`."""
    return _USERINFO_FALLBACK_PATTERN.sub(r"\g<scheme>://", text)


def scrub_credentials(text: str) -> str:
    """Drop URI user info, query and fragment, and mask password= style values. Idempotent."""
    text = _drop_free_text_userinfo(text)
    text = _FREE_TEXT_URI_PATTERN.sub(_scrub_uri_match, text)
    text = _PASSWORD_KEYWORD_PATTERN.sub(r"\g<keyword>\g<sep>***", text)
    return text


def redact_mapping(mapping: Mapping[Any, Any]) -> dict[Any, str]:
    """Keep every key, replace every value with ``'***'``."""
    return {key: "***" for key in mapping}


_MAX_RENDER_DEPTH = 32


def _redact_recursive(value: Any, ancestors: set[int]) -> Any:
    """Redact mappings and credential-shaped strings, walking lists/tuples/sets with a cycle guard."""
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, str):
        scrubbed = scrub_credentials(value)
        return value if scrubbed == value else scrubbed
    if isinstance(value, (list, tuple, set, frozenset)):
        if id(value) in ancestors:
            return "<cycle>"
        if len(ancestors) >= _MAX_RENDER_DEPTH:
            return "<...>"
        child_ancestors = ancestors | {id(value)}
        rendered = [_redact_recursive(item, child_ancestors) for item in value]
        if isinstance(value, tuple):
            return tuple(rendered)
        if isinstance(value, (set, frozenset)):
            return sorted(rendered, key=repr)
        return rendered
    return value


def redact_option_value(value: Any) -> Any:
    """Redact a Mapping, a credential-shaped string, or any of those nested in a list/tuple/set."""
    return _redact_recursive(value, set())
