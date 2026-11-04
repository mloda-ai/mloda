"""Strip credentials from free text before it reaches core's failure logs or MlodaRunError."""

import io
import re
from collections.abc import Iterable, Mapping
from contextvars import ContextVar, Token
from typing import Any
from urllib.parse import unquote

_AZURE_CONTAINER_SCHEMES = frozenset({"abfs", "abfss", "wasb", "wasbs"})
_AZURE_CONTAINER_PATTERN = re.compile(r"[a-z0-9-]{3,63}|\$(?:root|web|logs)")

# The lookbehind (not a bare \b) avoids quadratic backtracking on long alphanumeric runs; the jdbc
# subprotocol group is bounded (up to four words after `jdbc:`, e.g. `jdbc:h2:tcp`) for the same reason.
_SCHEME = r"(?<![A-Za-z0-9+.-])(?P<scheme>(?:jdbc:(?:[A-Za-z][A-Za-z0-9+.-]*:){0,3})?[A-Za-z][A-Za-z0-9+.-]*)"
_FREE_TEXT_URI_PATTERN = re.compile(
    _SCHEME + r"://"
    r"(?:(?P<userinfo>[^/?#\s]*)@)?"
    r"(?P<host>(?:\[[0-9A-Fa-f:.]+\]|[\w.~-]*)(?::\d+)?)"
    # `;` parameters directly after a pathless host (stops at `/`, so the path group still owns
    # later segments, and at `::`, so a chained URI keeps its own match).
    r"(?P<params>;(?:(?!::)[^\s?#\"'<>\]/])*)?"
    r"(?P<path>/[^\s?#\"'<>\]]*)?"
    r"(?:[?#][^\s\"'<>)\]]*)?"
)

_USERINFO_HEAD = _SCHEME + r"://[^\s/?#@:]*:(?!\d+(?:[?#\s]|$|"
_USERINFO_TAIL = r"))(?:(?!://)[^\s@])*@"
# strip_uri_userinfo feeds a data-access identity, which must name the host a client contacts: `host:port/path` is a host.
_USERINFO_FALLBACK_PATTERN = re.compile(_USERINFO_HEAD + r"/" + _USERINFO_TAIL)
# scrub_credentials reads `name:123/x@host/` as `user:pwd@` (a password), since free text has no such constraint.
# The guard after the slash also counts a bare `@host` at the end of the token (no path after the host)
# as that userinfo shape, so `name:12/x@host` is read the same way; a dotted or port-carrying tail
# (`a@b.csv`, `pkg@1.2.3`) stays a path, as the kept cases in the tests pin.
_FREE_TEXT_USERINFO_PATTERN = re.compile(
    _USERINFO_HEAD + r"/(?![^\s/?#@]+@(?:[\w.~-]+(?::\d+)?/|[\w~-]+(?=[?#\s]|$)))" + _USERINFO_TAIL
)

# Shared suffix a credential-shaped key name must end with, reused by the keyword and quoted-key patterns.
_SECRET_NAME = (  # nosec B105
    r"(?:password|passwd|pwd|secret|token|signature|sig|api[_-]?key|access[_-]?key|"
    r"account[_-]?key|secret[_-]?key|private[_-]?key|sessionid|sessid)"
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
# Slash in userinfo right after `://`; the dotless user keeps `unpkg.com/react@18/` and `github.com/user@host` intact.
_SCHEME_SLASH_DSN_PATTERN = re.compile(r"(?<=://)(?P<user>[\w$#~-]+)/[^\s/@]+@(?=[\w.-]+(?:[/:?#\s)\]'\"]|$))")

_PLAIN_URI_SCHEMES = (
    "http|https|s3|s3a|s3n|gs|gcs|abfs|abfss|wasb|wasbs|az|adl|file|ftp|sftp|hdfs|"
    "postgres|postgresql|mysql|mongodb|redis|sqlite|snowflake|mssql|ws|wss|ssh|git|grpc|kafka|ftps|"
    "oracle|redshift|bigquery|databricks"
)
# `user:password://host`: the password sits in the scheme position. Delimiters are excluded from both classes
# and the lookbehind anchors the start, so the scan stays linear. The password may hold `@` and be followed by one
# `:word`; a `/` may precede the user. `jdbc:` is never the user or the head of the password.
_DELIM = "\\s\"'(\\[{<=,;"
_PLAIN_SCHEME_GUARD = rf"(?!(?i:{_PLAIN_URI_SCHEMES})(?:\+[\w.-]+)?://)"
_PASSWORD_AS_SCHEME_PATTERN = re.compile(
    rf"(?<![^{_DELIM}/])(?!(?i:jdbc):)(?P<user>[^{_DELIM}:/@]+):"
    rf"(?!(?i:jdbc):){_PLAIN_SCHEME_GUARD}(?P<pw>[^{_DELIM}:/]+(?::{_PLAIN_SCHEME_GUARD}[^{_DELIM}:/@]+)?)(?=://)"
)


# Non-whitespace text after a jdbc match, cut at the next `://` so repeated URIs scan linearly.
_JDBC_TAIL_PATTERN = re.compile(r"(?:(?!://)\S)*")
# Private-use marker _uri_projection appends after a dropped jdbc path; _scrub_passes cuts it with the token tail.
# _scrub_uri_match appends the same marker when a dropped query or key=value path may continue past the match.
_JDBC_CUT = "\ue001"
_JDBC_CUT_PATTERN = re.compile(_JDBC_CUT + _JDBC_TAIL_PATTERN.pattern)
# Characters the free-text URI pattern refuses inside a query or fragment (a path refuses all but `)` too):
# a match ending right before one of them may have stopped in the middle of a dropped value.
_MATCH_END_TERMINATORS = frozenset("\"'<>)]")


def _cut_key_value_path(path: str) -> str:
    """The path up to (excluding) the first segment that looks like key=value after a ':'."""
    segments = path.split("/")
    for index, segment in enumerate(segments):
        if "=" in segment.partition(":")[2]:
            return "/".join(segments[:index])
    return path


def _uri_projection(match: re.Match[str]) -> str | None:
    scheme, userinfo, host = match["scheme"], match["userinfo"], match["host"]
    # Host parameters ride along with the path: the same key=value and hidden-percent checks apply.
    path = (match["params"] or "") + (match["path"] or "")
    container = ""
    if userinfo and scheme.lower() in _AZURE_CONTAINER_SCHEMES and _AZURE_CONTAINER_PATTERN.fullmatch(userinfo):
        container = f"{userinfo}@"
    if scheme.startswith("jdbc:"):
        tail = _JDBC_TAIL_PATTERN.search(match.string, match.end())
        if any(mark in path + (tail[0] if tail else "") for mark in "=;@"):
            return f"{scheme}://{container}{host}{_JDBC_CUT}"
    if any("=" in segment.partition(":")[2] for segment in path.split("/")):
        return None
    return f"{scheme}://{container}{host}{_cut_hidden_percent(scheme, host, path)}"


def _cut_hidden_percent(scheme: str, host: str, path: str) -> str:
    """Keep a percent-encoded path unless its decoded form hides a credential; then cut at the first escape."""
    if "%" not in path:
        return path
    decoded = unquote(path)
    probe = f"{scheme}://{host}{decoded}"
    hides = "%" in decoded or any(mark in decoded for mark in ("@", "'", '"', "://", *_REGISTERED_SECRETS.get()))
    if not hides and _scrub_passes(probe) == probe:
        return path
    head = path.partition("%")[0]
    return head[: head.rfind("/") + 1]


def _query_part(match: re.Match[str]) -> str:
    """The query/fragment part of the matched text (from the first `?` or `#`), or empty."""
    starts = [index for index in (match[0].find("?"), match[0].find("#")) if index != -1]
    return match[0][min(starts) :] if starts else ""


def _cut_dropped_tail(match: re.Match[str], scrubbed: str, dropped_key_value: bool) -> str:
    """Append the cut marker when a dropped key=value value may continue past the end of the match.

    The URI pattern stops at a quote, bracket or angle bracket, so a dropped query or key=value path
    whose value contains such a character leaves the rest of the value in the text after the match.
    The marker makes _scrub_passes cut the token tail too, as it already does for a dropped jdbc path.
    A clean URL drops nothing, so its closing quote or bracket in the text survives.
    """
    if scrubbed.endswith(_JDBC_CUT) or not dropped_key_value:
        return scrubbed
    if match.end() < len(match.string) and match.string[match.end()] in _MATCH_END_TERMINATORS:
        return scrubbed + _JDBC_CUT
    return scrubbed


def _scrub_uri_match(match: re.Match[str]) -> str:
    scheme, host = match["scheme"], match["host"]
    path = (match["params"] or "") + (match["path"] or "")
    projection = _uri_projection(match)
    if projection is not None:
        return _cut_dropped_tail(match, projection, "=" in _query_part(match))
    scrubbed = f"{scheme}://{host}{_cut_hidden_percent(scheme, host, _cut_key_value_path(path))}"
    return _cut_dropped_tail(match, scrubbed, True)


def _drop_free_text_userinfo(text: str, pattern: re.Pattern[str] = _FREE_TEXT_USERINFO_PATTERN) -> str:
    """Fallback for a userinfo-shaped `scheme://name:pwd@` prefix; drops through the first non-port `@`."""
    return pattern.sub(r"\g<scheme>://", text)


_USER_ONLY_PATTERN = re.compile(_SCHEME + r"://(?P<userinfo>[^/?#\s@]*)@")


def _drop_user_only(match: re.Match[str]) -> str:
    scheme, userinfo = match["scheme"], match["userinfo"]
    if scheme.lower() in _AZURE_CONTAINER_SCHEMES and _AZURE_CONTAINER_PATTERN.fullmatch(userinfo):
        return match[0]
    return f"{scheme}://"


def strip_uri_userinfo(text: str) -> str:
    """Drop `user:pwd@` and `user@` userinfo after every `scheme://`, keeping an Azure container."""
    return _USER_ONLY_PATTERN.sub(_drop_user_only, _drop_free_text_userinfo(text, _USERINFO_FALLBACK_PATTERN))


_MAX_DECODE_ROUNDS = 4
_PART_SPLIT = re.compile(r"(/|::)")
_PARAMETER_REGION = re.compile(r"[;?#].", re.DOTALL)


def _decode_to_fixpoint(part: str) -> str | None:
    decoded = part
    for _ in range(_MAX_DECODE_ROUNDS + 1):
        nxt = unquote(decoded)
        if nxt == decoded:
            return decoded
        decoded = nxt
    return None


def _clean_head(part: str) -> str:
    candidates = [i for i, char in enumerate(part) if char in ";%"]
    low, high = 0, len(candidates)
    while low < high:  # cleanliness is monotone: a dirty prefix keeps every longer one dirty
        mid = (low + high) // 2
        decoded = _decode_to_fixpoint(part[: candidates[mid]])
        if decoded is None or any(mark in decoded for mark in ";?#"):
            high = mid
        else:
            low = mid + 1
    return part[: candidates[low - 1]] if low else ""


def _cut_path_parameter_regions(uri: str) -> str:
    """Cut every `/`- or `::`-separated part whose decoded form holds `;`, `?` or `#` plus text to head + `;***`."""
    parts = _PART_SPLIT.split(uri)
    for i in range(0, len(parts), 2):
        decoded = _decode_to_fixpoint(parts[i])
        if decoded is None or _PARAMETER_REGION.search(decoded):
            parts[i] = _clean_head(parts[i]) + ";***"
    return "".join(parts)


_REGISTERED_SECRETS: ContextVar[frozenset[str]] = ContextVar("mloda_registered_secrets", default=frozenset())
_IDENTITY_SECRETS: ContextVar[frozenset[str]] = ContextVar("mloda_identity_secrets", default=frozenset())
_REGISTERED_ESCAPED: ContextVar[frozenset[str]] = ContextVar("mloda_registered_escaped", default=frozenset())


def _with_escaped(literals: frozenset[str]) -> frozenset[str]:
    """The literals plus their repr-escaped forms, which an object's repr can carry instead."""
    return literals | {repr(literal)[1:-1] for literal in literals if literal}


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


def mask_literals(text: str, literals: Iterable[str]) -> str:
    """Replace every occurrence of each literal with ``***``, merging overlaps."""
    spans = _literal_spans(text, literals)
    if not spans and type(text) is str:
        return text
    out = io.StringIO()
    cursor = 0
    for start, end in _merge_spans(spans):
        out.write(text[cursor:start])
        out.write("***")
        cursor = end
    out.write(text[cursor:])
    return out.getvalue()


class registered_secrets_scope:  # noqa: N801
    """Add literals to the set ``scrub_credentials`` masks exactly, for this context only.

    ``identity`` is the set that masks a published data_access_identity; None means the same as ``literals``.
    A class, not @contextmanager: that helper assigns ``__traceback__``, which a frozen-dataclass exception rejects.
    """

    def __init__(self, literals: frozenset[str], identity: frozenset[str] | None = None) -> None:
        self._literals = literals
        self._identity = literals if identity is None else identity
        self._tokens: list[tuple[Token[frozenset[str]], Token[frozenset[str]], Token[frozenset[str]]]] = []
        self._tracked: list[int | None] = []

    def __enter__(self) -> None:
        previous = _REGISTERED_SECRETS.get()
        merged = previous | self._literals
        identity = _IDENTITY_SECRETS.get() | self._identity
        escaped = _REGISTERED_ESCAPED.get() | _with_escaped(self._literals)
        self._tokens.append(
            (
                _REGISTERED_SECRETS.set(merged),
                _IDENTITY_SECRETS.set(identity),
                _REGISTERED_ESCAPED.set(escaped),
            )
        )
        if merged and not previous:
            from mloda.core.abstract_plugins.components.log_masking import ensure_log_masking

            ensure_log_masking()
        from mloda.core.abstract_plugins.components.log_masking import _track_scope_literals

        self._tracked.append(_track_scope_literals(self._literals))

    def __exit__(self, *exc_info: object) -> None:
        generation = self._tracked.pop()
        if generation is not None:
            from mloda.core.abstract_plugins.components import log_masking

            if generation == log_masking._fork_generation:
                log_masking._release_scope_literals(self._literals)
        secrets_token, identity_token, escaped_token = self._tokens.pop()
        _REGISTERED_ESCAPED.reset(escaped_token)
        _IDENTITY_SECRETS.reset(identity_token)
        _REGISTERED_SECRETS.reset(secrets_token)


def current_registered_secrets() -> frozenset[str]:
    return _REGISTERED_SECRETS.get()


def current_registered_escaped() -> frozenset[str]:
    return _REGISTERED_ESCAPED.get()


def current_identity_secrets() -> frozenset[str]:
    return _IDENTITY_SECRETS.get()


def scrub_credentials(text: str) -> str:
    """Mask URI user info, bearer/Authorization tokens, key=value and quoted-key secrets, user/password@host DSNs.

    Patterns run on the original text so a literal never deletes their anchors; every output char that came
    from a registered literal is masked, so no fragment of one survives.
    """
    registered = _REGISTERED_SECRETS.get()
    if not registered:
        return _scrub_passes(text)
    hidden = _merge_spans(_literal_spans(text, registered))
    scrubbed, origin = _scrub_tracked(text)
    if hidden:
        scrubbed = _mask_hidden(scrubbed, origin, hidden, len(text))
    return mask_literals(scrubbed, registered)


def _mask_hidden(text: str, origin: list[int], hidden: list[tuple[int, int]], size: int) -> str:
    flags = bytearray(size)
    for start, end in hidden:
        flags[start:end] = b"\x01" * (end - start)
    out = io.StringIO()
    tail = ""  # last three emitted chars: a masked run right after `***` emits nothing
    in_run = False
    for index, source in enumerate(origin):
        if source >= 0 and flags[source]:
            in_run = True
            continue
        if in_run and tail != "***":
            out.write("***")
            tail = "***"
        in_run = False
        out.write(text[index])
        tail = (tail + text[index])[-3:]
    if in_run and tail != "***":
        out.write("***")
    return out.getvalue()


def _replacement(match: re.Match[str], repl: Any) -> str:
    return str(repl(match)) if callable(repl) else match.expand(repl)


def _align(matched: str, matched_origin: list[int], replacement: str) -> list[int]:
    """Origins for a replacement: greedy subsequence walk of its non-`*` chars through the matched text."""
    result: list[int] = []
    position = 0
    for char in replacement:
        found = position
        if char != "*":
            while found < len(matched) and matched[found] != char:
                found += 1
        if char != "*" and found < len(matched):
            result.extend((matched_origin[found],))
            position = found + 1
        else:
            result.extend((-1,))
    return result


def _sub_tracked(pattern: re.Pattern[str], repl: Any, text: str, origin: list[int]) -> tuple[str, list[int]]:
    out = io.StringIO()
    origins: list[int] = []
    cursor = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        out.write(text[cursor:start])
        origins.extend(origin[cursor:start])
        replacement = _replacement(match, repl)
        out.write(replacement)
        origins.extend(_align(match[0], origin[start:end], replacement))
        cursor = end
    out.write(text[cursor:])
    origins.extend(origin[cursor:])
    return out.getvalue(), origins


def _authorization_repl(m: re.Match[str]) -> str:
    return m["head"] + (f"{m['scheme']} ***" if m["scheme"] else "***")


_QUOTED_KEY_REPL = r"\g<lead>\g<quote>\g<keyword>\g<quote>: \g<quote>***\g<quote>"

_PASSES: list[tuple[re.Pattern[str], Any]] = [
    (_FREE_TEXT_USERINFO_PATTERN, r"\g<scheme>://"),
    (_FREE_TEXT_URI_PATTERN, _scrub_uri_match),
    (_JDBC_CUT_PATTERN, ""),  # a dropped jdbc path also takes its token tail
    (_PASSWORD_AS_SCHEME_PATTERN, r"\g<user>:***"),
    (_AUTHORIZATION_PATTERN, _authorization_repl),
    (_BEARER_PATTERN, r"\g<scheme> ***"),
    (_KEYWORD_PATTERN, r"\g<keyword>\g<sep>***"),
    (_QUOTED_KEY_PATTERN, _QUOTED_KEY_REPL),
    (_SLASH_DSN_PATTERN, r"\g<user>/***@"),
    (_SCHEME_SLASH_DSN_PATTERN, r"\g<user>/***@"),
]


def _scrub_passes(text: str) -> str:
    text = text.replace(_JDBC_CUT, "\ufffd")
    for pattern, repl in _PASSES:
        text = pattern.sub(repl, text)
    return text


def _scrub_tracked(text: str) -> tuple[str, list[int]]:
    text = text.replace(_JDBC_CUT, "\ufffd")
    origin = list(range(len(text)))
    for pattern, repl in _PASSES:
        text, origin = _sub_tracked(pattern, repl, text, origin)
    return text, origin


# Bounds recursion so a deeply nested attribute cannot hit RecursionError on the failure path.
_MAX_MASK_DEPTH = 32


def _mask_item(item: Any, literals: frozenset[str], memo: dict[int, Any] | None = None, depth: int = 0) -> Any:
    """Mask str leaves in tuples, lists and dicts; return ``item`` itself when nothing inside changed."""
    if isinstance(item, str):
        masked_str = mask_literals(item, literals)
        return item if masked_str == item else masked_str
    if not isinstance(item, (tuple, list, dict)):
        return item
    if depth > _MAX_MASK_DEPTH:
        return "***"
    memo = {} if memo is None else memo
    if id(item) in memo:
        return memo[id(item)]
    if isinstance(item, list):
        masked_list: list[Any] = []
        memo[id(item)] = masked_list
        masked_list.extend(_mask_item(element, literals, memo, depth + 1) for element in item)
        if all(new is old for new, old in zip(masked_list, item)):
            memo[id(item)] = item
            return item
        return masked_list
    if isinstance(item, dict):
        masked_dict: dict[Any, Any] = {}
        memo[id(item)] = masked_dict
        unchanged = True
        for key, element in item.items():
            new_key = _mask_item(key, literals, memo, depth + 1)
            new_element = _mask_item(element, literals, memo, depth + 1)
            masked_dict[new_key] = new_element
            unchanged = unchanged and new_key is key and new_element is element
        if unchanged:
            memo[id(item)] = item
            return item
        return masked_dict
    memo[id(item)] = "***"
    elements = [_mask_item(element, literals, memo, depth + 1) for element in item]
    if all(new is old for new, old in zip(elements, item)):
        memo[id(item)] = item
        return item
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
        masked_args = _mask_item(current.args, literals)
        if masked_args is not current.args:
            object.__setattr__(current, "args", masked_args)
        attributes = vars(current)
        for name, value in list(attributes.items()):
            if isinstance(value, (str, dict, list, tuple)) and not name.startswith("__"):
                masked = _mask_item(value, literals)
                if masked is not value:
                    attributes[name] = masked
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
