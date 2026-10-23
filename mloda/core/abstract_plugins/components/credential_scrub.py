"""Strip credentials from free text before it reaches core's failure logs or MlodaRunError."""

import re
from collections.abc import Iterable, Mapping
from contextvars import ContextVar, Token
from typing import Any

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

# Shared suffix a credential-shaped key name must end with, reused by the keyword and quoted-key patterns.
_SECRET_NAME = (  # nosec B105
    r"(?:password|passwd|pwd|secret|token|signature|sig|api[_-]?key|access[_-]?key|"
    r"account[_-]?key|secret[_-]?key|private[_-]?key)"
)

_KNOWN_SCHEMES = r"(?i:bearer|basic|token|digest|apikey|negotiate)"

# (?<![\w-]), not (?<![A-Za-z]): a name-start anchor so a prefixed identifier (sslpassword,
# PGPASSWORD, X-Api-Key) still matches, captured whole so the prefix survives the replacement.
# Optional scheme prefix so `access_token=Bearer abc` masks the token, not just the word `Bearer`.
# Trailing `}` optional: an unterminated `{` masks to end of line instead of rescanning.
_KEYWORD_PATTERN = re.compile(
    rf"(?<![\w-])(?P<keyword>[\w-]*{_SECRET_NAME})(?P<sep>[ \t]*=[ \t]*)"
    rf"(?:{_KNOWN_SCHEMES}[ \t]+)?"
    r"(?:'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"|\{(?:[^}\n]|\}\})*\}?|[^;&\s'\"]*)",
    re.IGNORECASE,
)


def _container_body(closer: str, depth: int = 0) -> str:
    """Quote-aware container body: a closer inside a quoted string does not end it."""
    quoted = r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\""
    if depth == 0:
        return rf"(?:{quoted}|[^\{closer}\[{{(\"'\n])*(?:[\[{{(][^\n]*)?"
    inner = depth - 1
    nested = (
        rf"\[{_container_body(']', inner)}\]?|\{{{_container_body('}', inner)}\}}?|\({_container_body(')', inner)}\)?"
    )
    return rf"(?:{quoted}|{nested}|[^\{closer}\[{{(\"'\n])*"


# Nesting levels inside the outer container; pattern size grows ~3^depth, keep <= 4.
# Values nested deeper mask to end of line (fail closed).
_NEST_DEPTH = 3


# Anchored on a `{`/`,` lead so prose like `invalid password: too short` or unquoted `password: X` is left alone.
_QUOTED_KEY_PATTERN = re.compile(
    r"(?P<lead>[{,][ \t]*)(?P<quote>['\"])(?P<keyword>[\w-]*" + _SECRET_NAME + r")(?P=quote)[ \t]*:[ \t]*"
    r"(?:b?'(?:[^'\\\n]|\\.)*'|b?\"(?:[^\"\\\n]|\\.)*\""
    rf"|\[{_container_body(']', _NEST_DEPTH)}\]?|\{{{_container_body('}', _NEST_DEPTH)}\}}?"
    rf"|(?:[A-Za-z_][\w.]*)?\({_container_body(')', _NEST_DEPTH)}\)?|[^,}}\s]*)",
    re.IGNORECASE,
)

# Token-shaped (not prose): has a digit or one of -_~+/, a `.` before a token char, or is 20+ chars long.
_TOKEN68 = r"[A-Za-z0-9\-._~+/]"
_TOKEN_SHAPE = rf"(?={_TOKEN68}{{20}}|{_TOKEN68}*(?:[0-9\-_~+/]|\.{_TOKEN68}))"
_BEARER_PATTERN = re.compile(rf"(?<![\w-])(?P<scheme>(?i:bearer))[ \t]+{_TOKEN_SHAPE}{_TOKEN68}+=*")

# A scheme word is known or scheme-shaped (not all lowercase letters, so prose like `denied for` is skipped);
# the rest of the value is masked up to the closing quote when the head opened one, else to end of line.
_AUTHORIZATION_PATTERN = re.compile(
    r"(?P<head>(?<![\w-])(?i:(?:(?:x|http|proxy)[-_])*authorization(?:[-_]header)?)['\"]?[ \t]*[:=][ \t]*(?P<q>['\"])?)"
    rf"(?:(?P<scheme>{_KNOWN_SCHEMES}|(?![a-z]+(?![\w-]))[A-Za-z][\w-]*)[ \t]+"
    r"(?(q)(?:(?!(?P=q))[^\n])*|[^\n]*)"
    rf"|{_TOKEN_SHAPE}{_TOKEN68}+=*)"
)

# Oracle-style slash DSN: user/password@host, distinguished from a lookalike path by the trailing
# host shape (EZConnect //, a descriptor (, or host:port) so `path/to/x@y` and `user@example.com` are left alone.
# (?!\d+/) keeps a bare port like `8080/` from being read as the user.
_SLASH_DSN_PATTERN = re.compile(
    r"(?<![\w.$#/-])(?!\d+/)(?P<user>[\w.$#-]+)/[^\s/]+@(?=//|\(|[\w.-]+:\d+(?!\w)|[\w.-]+/\w)"
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


_USER_ONLY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+.-])(?P<scheme>(?:jdbc:)?[A-Za-z][A-Za-z0-9+.-]*)://(?P<userinfo>[^/?#\s@]*)@"
)


def _drop_user_only(match: re.Match[str]) -> str:
    scheme, userinfo = match["scheme"], match["userinfo"]
    if scheme.lower() in _AZURE_CONTAINER_SCHEMES and _AZURE_CONTAINER_PATTERN.fullmatch(userinfo):
        return match[0]
    return f"{scheme}://"


def strip_uri_userinfo(text: str) -> str:
    """Drop `user:pwd@` and `user@` userinfo after every `scheme://`, keeping an Azure container."""
    return _USER_ONLY_PATTERN.sub(_drop_user_only, _drop_free_text_userinfo(text))


_PLACEHOLDER = "\ue000"
_REGISTERED_SECRETS: ContextVar[frozenset[str]] = ContextVar("mloda_registered_secrets", default=frozenset())


def _literal_spans(text: str, literals: Iterable[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for literal in literals:
        start = 0
        while True:
            index = text.find(literal, start)
            if index == -1:
                break
            spans.append((index, index + len(literal)))
            start = index + 1
    return spans


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def placeholder_mask(text: str, literals: Iterable[str]) -> str:
    """Replace every occurrence of each literal with an inert placeholder."""
    masked = ""
    cursor = 0
    for start, end in _merge_spans(_literal_spans(text, literals)):
        masked += text[cursor:start] + _PLACEHOLDER
        cursor = end
    return masked + text[cursor:]


def mask_literals(text: str, literals: Iterable[str]) -> str:
    """Replace every occurrence of each literal with ``***``, merging overlaps."""
    return placeholder_mask(text, literals).replace(_PLACEHOLDER, "***")


class registered_secrets_scope:  # noqa: N801
    """Add literals to the set ``scrub_credentials`` masks exactly, for this context only.

    A class, not @contextmanager: that helper assigns ``__traceback__``, which a frozen-dataclass exception rejects.
    """

    def __init__(self, literals: frozenset[str]) -> None:
        self._literals = literals
        self._tokens: list[Token[frozenset[str]]] = []

    def __enter__(self) -> None:
        self._tokens.append(_REGISTERED_SECRETS.set(_REGISTERED_SECRETS.get() | self._literals))

    def __exit__(self, *exc_info: object) -> None:
        _REGISTERED_SECRETS.reset(self._tokens.pop())


def current_registered_secrets() -> frozenset[str]:
    return _REGISTERED_SECRETS.get()


def scrub_credentials(text: str) -> str:
    """Mask URI user info, bearer/Authorization tokens, key=value and quoted-key secrets, user/password@host DSNs.

    Inside a ``registered_secrets_scope`` the registered literals are masked exactly first.
    """
    registered = _REGISTERED_SECRETS.get()
    if registered:
        text = placeholder_mask(text, registered)
    text = _drop_free_text_userinfo(text)
    text = _FREE_TEXT_URI_PATTERN.sub(_scrub_uri_match, text)
    text = _AUTHORIZATION_PATTERN.sub(lambda m: m["head"] + (f"{m['scheme']} ***" if m["scheme"] else "***"), text)
    text = _BEARER_PATTERN.sub(r"\g<scheme> ***", text)
    text = _KEYWORD_PATTERN.sub(r"\g<keyword>\g<sep>***", text)
    text = _QUOTED_KEY_PATTERN.sub(r"\g<lead>\g<quote>\g<keyword>\g<quote>: \g<quote>***\g<quote>", text)
    text = _SLASH_DSN_PATTERN.sub(r"\g<user>/***@", text)
    if registered:
        text = text.replace(_PLACEHOLDER, "***")
    return text


def _mask_item(item: Any, literals: frozenset[str], memo: dict[int, Any] | None = None) -> Any:
    """Mask str leaves, recursing into tuples, lists and dicts (types kept, cycles terminate via ``memo``)."""
    if isinstance(item, str):
        return mask_literals(item, literals)
    if not isinstance(item, (tuple, list, dict)):
        return item
    memo = {} if memo is None else memo
    if id(item) in memo:
        return memo[id(item)]
    if isinstance(item, list):
        masked_list: list[Any] = []
        memo[id(item)] = masked_list
        masked_list.extend(_mask_item(element, literals, memo) for element in item)
        return masked_list
    if isinstance(item, dict):
        masked_dict: dict[Any, Any] = {}
        memo[id(item)] = masked_dict
        for key, element in item.items():
            masked_dict[_mask_item(key, literals, memo)] = _mask_item(element, literals, memo)
        return masked_dict
    memo[id(item)] = item
    elements = [_mask_item(element, literals, memo) for element in item]
    result = item._make(elements) if hasattr(item, "_make") else tuple(elements)
    memo[id(item)] = result
    return result


def redact_exception(exc: BaseException, literals: frozenset[str]) -> BaseException:
    """Mask literals in place across the exception, its chain, group members, notes and refusals."""
    if not literals:
        return exc
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        object.__setattr__(current, "args", _mask_item(tuple(current.args), literals))
        for name, value in list(vars(current).items()):
            if isinstance(value, str) and not name.startswith("__"):
                object.__setattr__(current, name, mask_literals(value, literals))
        if isinstance(current, OSError):
            for name in ("strerror", "filename", "filename2"):
                object.__setattr__(current, name, _mask_item(getattr(current, name), literals))
        notes = getattr(current, "__notes__", None)
        if isinstance(notes, list):
            object.__setattr__(current, "__notes__", [_mask_item(note, literals) for note in notes])
        members: list[Any] = []
        for name in ("exceptions", "refusals"):
            value = getattr(current, name, None)
            if isinstance(value, (tuple, list)):
                members.extend(value)
        pending.extend(
            nested for nested in (current.__cause__, current.__context__, *members) if isinstance(nested, BaseException)
        )
    return exc


class redact_escaping_exception:  # noqa: N801
    """Mask the current registered literals in any exception leaving the block (a class: see registered_secrets_scope)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        if exc is not None:
            redact_exception(exc, current_registered_secrets())


def redact_mapping(mapping: Mapping[Any, Any]) -> dict[Any, str]:
    """Keep every key (scrubbing a credential-shaped str key), replace every value with ``'***'``."""
    result: dict[Any, str] = {}
    for key in mapping:
        if isinstance(key, str):
            scrubbed = scrub_credentials(key)
            result[key if scrubbed == key else scrubbed] = "***"
        else:
            result[key] = "***"
    return result


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


_SECRET_KEY = re.compile(r"[\w-]*" + _SECRET_NAME, re.IGNORECASE)


def _mask_secret_keys(value: Any, ancestors: frozenset[int] = frozenset()) -> Any:
    """Replace the value under every credential-shaped key with ``'***'``, walking mappings, lists and tuples."""
    from mloda.core.abstract_plugins.components.credential import RegisteredCredential  # credential imports us

    if isinstance(value, RegisteredCredential):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        if id(value) in ancestors:
            return "<cycle>"
        inner = ancestors | {id(value)}
    if isinstance(value, Mapping):
        return {
            key: "***" if isinstance(key, str) and _SECRET_KEY.fullmatch(key) else _mask_secret_keys(item, inner)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [_mask_secret_keys(item, inner) for item in value]
        if hasattr(value, "_make"):
            return value._make(items)
        return tuple(items) if isinstance(value, tuple) else items
    return value


def redact_option_value(value: Any) -> Any:
    """Redact a Mapping, a credential-shaped string, or any of those nested in a list/tuple/set."""
    return _redact_recursive(value, set())
