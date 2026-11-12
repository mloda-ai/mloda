"""Strip credentials from free text before it reaches core's failure logs or MlodaRunError."""

import io
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from contextvars import ContextVar, Token
from typing import Any
from urllib.parse import unquote

_OBJECT_STORE_SCHEMES = frozenset({"s3", "s3a", "s3n", "gs", "gcs", "abfs", "abfss", "wasb", "wasbs", "az", "adl"})
_PWD_ROOTS = frozenset(
    {
        "home",
        "Users",
        "root",
        "tmp",
        "var",
        "opt",
        "srv",
        "usr",
        "mnt",
        "workspace",
        "workspaces",
        "app",
        "data",
        "builds",
        "runner",
    }
)
_AZURE_CONTAINER_SCHEMES = frozenset({"abfs", "abfss", "wasb", "wasbs"})
_AZURE_CONTAINER_PATTERN = re.compile(r"[a-z0-9-]{3,63}|\$(?:root|web|logs)")

# The lookbehind (not a bare \b) avoids quadratic backtracking on long alphanumeric runs; the jdbc
# subprotocol group is bounded (up to four words after `jdbc:`, e.g. `jdbc:h2:tcp`) for the same reason.
_SCHEME = r"(?<![A-Za-z0-9+.-])(?P<scheme>(?:(?i:jdbc):(?:[A-Za-z][A-Za-z0-9+.-]*:){0,3})?[A-Za-z][A-Za-z0-9+.-]*)"
# A `::` that continues an fsspec chain: before a `<scheme>://` or a lowercase `<protocol>::` element.
_CHAIN_COLONS = r"::(?=[A-Za-z][A-Za-z0-9+.-]*://|[a-z][a-z0-9+.-]*::)"
# A `::` after a value continues a chain only when its protocol run reaches a `scheme://`; the run is bounded so the
# lookahead is constant work, and a longer run reads as a value tail and is cut.
_CHAIN_AFTER_VALUE = r"::(?=(?:[a-z][a-z0-9+.-]{0,31}::){0,3}[A-Za-z][A-Za-z0-9+.-]*://)"
_FREE_TEXT_URI_PATTERN = re.compile(
    _SCHEME + r"://"
    r"(?:(?P<userinfo>[^/?#\s]*)@)?"
    r"(?P<host>(?:\[[0-9A-Fa-f:.]+\]|[\w.~=-]*)(?::\d+)?)"
    # `;` or percent-escape parameters directly after a pathless host (stops at `/`, so the path group still owns
    # later segments, at a chain `::`, so a chained URI keeps its match; a `;` before another URI is no parameter).
    rf"(?P<params>(?:(?:;(?!(?:(?i:jdbc):(?:[A-Za-z][A-Za-z0-9+.-]*:){{0,3}})?[A-Za-z][A-Za-z0-9+.-]*://)|(?=%))(?:(?!{_CHAIN_COLONS})[^\s?#\"'<>\];/])*)+)?"
    # Stop at `;scheme://` so `a.csv;s3://b/b.csv` is two URIs.
    r"(?P<path>/(?:[^\s?#\"'<>\];]|;(?![A-Za-z][A-Za-z0-9+.-]*://))*)?"
    r"(?P<query>[?#][^\s\"'<>)\]]*)?"
)

_USERINFO_HEAD = _SCHEME + r"://[^\s/?#@:]*:(?!\d+(?:[?#\s]|$|"
_USERINFO_TAIL = r"))(?:(?!://)[^\s@])*@"
# strip_uri_userinfo feeds a data-access identity, which must name the host a client contacts: `host:port/path` is a host.
_USERINFO_FALLBACK_PATTERN = re.compile(_USERINFO_HEAD + r"/" + _USERINFO_TAIL)
# scrub_credentials reads `name:123/x@host/` as `user:pwd@` (a password), since free text has no such constraint;
# that includes `https://host:443/user@example.com/profile`, kept on purpose (fail closed).
_FREE_TEXT_USERINFO_PATTERN = re.compile(_USERINFO_HEAD + r"/(?![^\s/?#@]+@[\w.~-]+(?::\d+)?/)" + _USERINFO_TAIL)
_DB_SCHEMES = (
    "postgres|postgresql|mysql|mariadb|mongodb|redis|rediss|mssql|sqlserver|oracle|snowflake|redshift|"
    "clickhouse|cockroachdb|amqp|amqps"
)
_DB_CREDENTIAL = r"(?:(?!://)[^\s@?#;&=,'\"])*"
_DB_HOST_TAIL = r"(?:[\w~-]+(?:\.[\w~-]+)+(?::\d+)?|[\w.~-]+:\d+)(?=[/?#\s\"')\]}>]|$)"
# `host:port/path@handle` is a source handle, so only a non-port digit run or a host-shaped tail reads as a password.
_DB_USERINFO_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9+.-])(?<!(?i:jdbc):)(?P<scheme>(?i:{_DB_SCHEMES})(?:\+[\w.-]+)?)://[^\s/?#@:]*:"
    rf"(?:\d{{1,3}}/{_DB_CREDENTIAL}@|\d+/{_DB_CREDENTIAL}@(?={_DB_HOST_TAIL}))"
)

# Shared suffix a credential-shaped key name must end with, reused by the keyword and quoted-key patterns.
_SECRET_NAME = (  # nosec B105
    r"(?:password|passwd|pwd|secret|token|signature|sig|api[._-]?key|access[._-]?key|"
    r"account[._-]?key|secret[._-]?key|private[._-]?key|sessionid|sessid)"
)

_KNOWN_SCHEMES = r"(?i:bearer|basic|token|digest|apikey|negotiate)"


def _quoted_value(quote: str, extra: str) -> str:
    """A quoted value (doubled quotes allowed inside); the glued non-delimiter run after its close stays part of it."""
    end = rf"[\s;&,:)\]}}>\\{extra}]|$|[.!?]+(?:\s|$|['\"])"
    glued = rf"[^;&\s'\"{extra}]++(?!{quote})"
    return rf"{quote}(?:[^{quote}\\\n]|\\.|{quote}{quote}(?!\s|$))*+{quote}(?:(?={end}|['\"](?:{end}))|{glued})"


# (?<![\w.-]), not (?<![A-Za-z]): a name-start anchor so a prefixed or dotted identifier (sslpassword,
# PGPASSWORD, X-Api-Key, fs.s3a.secret.key) still matches, captured whole so the prefix survives the replacement.
# Optional scheme prefix so `access_token=Bearer abc` masks the token, not just the word `Bearer`.
# Trailing `}` optional: an unterminated `{` masks to end of line instead of rescanning.
def _keyword_pattern(plain: str, extra: str = "") -> re.Pattern[str]:
    single, double = _quoted_value("'", extra), _quoted_value('"', extra)
    return re.compile(
        rf"(?<![\w.-])(?P<keyword>[\w.-]*{_SECRET_NAME})(?P<sep>[ \t]*=[ \t]*)"
        rf"(?P<known>{_KNOWN_SCHEMES}[ \t]+)?"
        rf"(?P<value>(?P<quoted>{single}|{double})"
        r"|(?P<braced>\{(?:[^}\n]|\}\})*\}?)|" + plain + ")",
        re.IGNORECASE,
    )


# A quote run continues the value unless a delimiter or the end follows it, so no fragment of the value is left.
_KEYWORD_PATTERN = _keyword_pattern(r"[^;&\s'\"]*(?:['\"]++(?![\s,;&:)\]}>]|$)[^;&\s'\"]+)*")
# The identity form: a segment value ends at `/` or at a `::` that continues a chain; an inner quote run continues it.
_KEYWORD_SEGMENT_PATTERN = _keyword_pattern(
    rf"(?:(?!{_CHAIN_AFTER_VALUE})(?:::(?:[a-z][a-z0-9+.-]{{0,31}}::)*+|[^;&/\s'\"]))*"
    rf"(?:['\"]++(?=[^;&/\s'\"])(?:(?!{_CHAIN_AFTER_VALUE})(?:::(?:[a-z][a-z0-9+.-]{{0,31}}::)*+|[^;&/\s'\"]))+)*",
    "/",
)
_PATH_VALUE = re.compile(r"/[\w.-]+(?:/[\w.-]+)+/?")
_COUNT_NAME = re.compile(r"(?:max|min|num|total)[_-]?token", re.IGNORECASE)


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


# The lead may carry a repr-escaped newline (`\n`, `\\n`).
# Anchored on a `{`/`,` lead so prose like `invalid password: too short` or unquoted `password: X` is left alone.
# A key quote may carry a backslash run (repr or JSON dump).
# A bare closing quote must follow a delimiter-free run; a quote after any backslash run is content.
# `:` is not a delimiter, or a value could close at the next key's quote and leave that key's secret unmatched.
_BARE_DELIMITER = r"""[\s,;}\])\\'"]|\Z"""
_BARE_STRING = (
    rf"b?'(?:[^'\\\n]|\\++['\"]?+|'(?!{_BARE_DELIMITER}))*+(?:'(?={_BARE_DELIMITER})|(?=\r?\n|\Z))"
    rf"|b?\"(?:[^\"\\\n]|\\++['\"]?+|\"(?!{_BARE_DELIMITER}))*+(?:\"(?={_BARE_DELIMITER})|(?=\r?\n|\Z))"
)
_QUOTED_KEY_PATTERN = re.compile(
    r"(?P<lead>[{,](?:\s|\\+[nrt])*)(?P<quote>(?P<esc>\\+)?['\"])(?P<keyword>[\w.-]*"
    + _SECRET_NAME
    + r")(?P=quote)[ \t]*:[ \t]*"
    rf"(?P<value>{_BARE_STRING}|(?P<vq>b?\\+['\"])"
    r"|(?(esc)(?:[A-Za-z_][\w.]*)?[\[{(]|(?:"
    rf"\[{_container_body(']', _NEST_DEPTH)}\]?|\{{{_container_body('}', _NEST_DEPTH)}\}}?"
    rf"|(?:[A-Za-z_][\w.]*)?\({_container_body(')', _NEST_DEPTH)}\)?))|[^,}}\s]*)",
    re.IGNORECASE,
)
_STRING_LITERAL = r"b?(?:'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\")"
_STRING_ONLY = re.compile(_STRING_LITERAL)
_CONTAINER_HEAD = re.compile(r"(?:[A-Za-z_][\w.]*)?[\[{(]")
_CONTINUATION_LINE = re.compile(
    rf"[ \t]*(?P<literal>{_STRING_LITERAL})[ \t,]*(?:(?P<closer>[\])}}])[\])}}, \t]*|(?=\r?\n|\Z))"
)
_ATOM = rf"(?:{_STRING_LITERAL}|[-+]?\.?\d[\w.+-]*|(?:true|false|null|True|False|None|NaN|-?Infinity)\b)"
_PAIR_OR_ATOM = rf"(?:{_STRING_LITERAL}[ \t]*:[ \t]*{_ATOM}|{_ATOM})"
_INLINE_CONTAINER = rf"[\[{{(][ \t]*(?:{_PAIR_OR_ATOM}(?:[ \t]*,[ \t]*{_PAIR_OR_ATOM})*[ \t,]*)?[\]}})]"
_MEMBER = rf"(?:{_PAIR_OR_ATOM}|{_INLINE_CONTAINER})"
_INLINE_CONTAINER2 = rf"[\[{{(][ \t]*(?:{_MEMBER}(?:[ \t]*,[ \t]*{_MEMBER})*[ \t,]*)?[\]}})]"
_ITEM = rf"(?:{_ATOM}|{_INLINE_CONTAINER2})"
# A whole bare-item line of a list body; text may follow once a closer was seen.
_ITEM_LINE = re.compile(rf"[ \t]*,?[ \t]*{_ITEM}(?:[ \t]*,[ \t]*{_ITEM})*[ \t,]*(?:[\])}}].*|(?=\r?\n|\Z))")
_PAIR = rf"{_STRING_LITERAL}[ \t]*:[ \t]*{_ITEM}"
_PAIR_LINE = re.compile(rf"[ \t]*,?[ \t]*{_PAIR}(?:[ \t]*,[ \t]*{_PAIR})*[ \t,]*(?:[\])}}}}].*|(?=\r?\n|\Z))")
# An item line ending in an open container head followed by comma-terminated items.
_OPEN_HEAD_LINE = re.compile(
    rf"[ \t]*,?[ \t]*(?:{_ITEM}[ \t]*,[ \t]*)*[\[{{(][ \t]*(?:(?:{_PAIR}|{_ITEM})[ \t]*,[ \t]*)*(?=\r?\n|\Z)"
)
_COMMA_AHEAD = re.compile(r"[ \t\r\n]*,")
_ESCAPED_NEWLINE = r"(?:\\+r)?\\+n"
_ESCAPED_CHUNK_HEAD = re.compile(rf"{_ESCAPED_NEWLINE}(?:[ \t]|\\+t)+(?=b?['\"]|b?\\+['\"])")
_ESCAPED_CHUNK_TAIL = re.compile(rf"[ \t,]*(?:(?P<closer>[\])}}])[\])}}, \t]*|(?={_ESCAPED_NEWLINE}|['\"]|\Z))")
_FLAT_BODY_LINE = re.compile(
    r"""b?['"]|[-+.]?\d|[\])},]|(?:true|false|null|True|False|None|NaN|-?Infinity)\b|[\[{(][ \t]*(?:[\]})][ \t,]*)?(?=\r?\n|\Z)"""
)
_QUOTED_KEY_REPL = r"\g<lead>\g<quote>\g<keyword>\g<quote>: \g<quote>***\g<quote>"


def _indent_end(text: str, start: int) -> int:
    while start < len(text) and text[start] in " \t":
        start += 1
    return start


def _consistent_close(text: str, position: int) -> bool:
    """Whether a delimiter, quote, backslash or line/text end follows the close."""
    after = _indent_end(text, position)
    return after >= len(text) or text[after] in ",:;]})\\'\"\r\n"


def _string_end(text: str, start: int, extra: int = 0, strict: bool = False) -> tuple[int, bool, bool]:
    """(end, closed, consistent) of the string opened at start; it closes at the next same quote whose
    backslash run r has r % 2**(bit_length(r0) + 1 + extra) == r0. Strict stops at the first inconsistent close."""
    size = len(text)
    body = start
    while body < size and text[body] == "\\":
        body += 1
    if body >= size:
        return size, False, True
    quote, opening = text[body], body - start
    period = 2 ** (opening.bit_length() + 1 + extra)
    run = 0
    for i in range(body + 1, size):
        char = text[i]
        if char == "\\":
            run += 1
        elif char == "\n":
            return i, False, True
        else:
            if char == quote and run % period == opening:
                if strict and not _consistent_close(text, i + 1):
                    return i + 1, False, False
                return i + 1, True, True
            run = 0
    return size, False, True


def _chosen_end(text: str, start: int, floor: int = 0) -> tuple[int, bool, int]:
    """(end, closed, extra): the first escape-level reading with a consistent close, never ending short of any scan."""
    reached = start
    for extra in (floor,) + tuple(e for e in (0, 1, 2) if e != floor):
        end, closed, consistent = _string_end(text, start, extra, True)
        reached = max(reached, end)
        if consistent:
            return reached, closed, extra
    end, closed, _ = _string_end(text, start)
    return max(reached, end), closed, 0


def _container_end(text: str, opener: int, key_indent: int) -> tuple[int, bool]:
    """(end, deeper) of a multi-line container value: its closer, else the line before the first shallow line.
    A flat body (not indented past its key) may pass value-like shallow lines.
    Item mode also passes shallow item, pair (in a dict), closer (nested) and open-head lines.
    deeper: a string was read at a deeper escape level, so the pattern's own quote reading is unreliable."""
    size = len(text)
    stack: list[str] = []
    flat: bool | None = None
    items = False
    seen_newline = False
    deeper = False
    level = 0
    i = opener
    while i < size:
        char = text[i]
        if char == "\\":
            j = i
            while j < size and text[j] == "\\":
                j += 1
            if j < size and text[j] in "'\"":
                i, _, extra = _chosen_end(text, i, level)
                deeper = deeper or extra > 0
                level = max(level, extra)
            else:
                i = j
            continue
        if char in "'\"":
            i, _, extra = _chosen_end(text, i, level)
            deeper = deeper or extra > 0
            level = max(level, extra)
            continue
        if char in "[{(":
            stack.append(char)
        elif char in "]})":
            if stack:
                stack.pop()
            if not stack:
                return i + 1, deeper
        elif char == "\n":
            if not seen_newline:
                seen_newline = True
                head = text[opener + 1 : i].rstrip(" \t\r")
                items = bool(head) and (head.endswith(",") or _COMMA_AHEAD.match(text, i + 1) is not None)
            first = _indent_end(text, i + 1)
            if first < size and text[first] not in "\r\n":
                shallow = first - (i + 1) <= key_indent
                if flat is None:
                    flat = shallow and text[_indent_end(text, opener + 1)] in "\r\n"
                if not shallow:
                    i += 1
                    continue
                if text[first] in "]})":
                    if len(stack) == 1:
                        return first + 1, deeper
                    i += 1
                    continue
                if (
                    (flat and (_FLAT_BODY_LINE.match(text, first) or _ITEM_LINE.match(text, first)))
                    or (items and _ITEM_LINE.match(text, first))
                    or (items and _OPEN_HEAD_LINE.match(text, first))
                    or (items and stack[-1] == "{" and _PAIR_LINE.match(text, first))
                ):
                    i += 1
                    continue
                stop = i - 1 if text[i - 1] == "\r" else i
                while not flat and stop > opener and text[stop - 1] in " \t,":
                    stop -= 1
                return stop, deeper
        i += 1
    return size, deeper


def _string_continuation_end(text: str, end: int, key_indent: int) -> int:
    """Extend a closed string value over following deeper lines made of one string literal each."""
    size = len(text)
    position = end
    while position < size:
        if text[position] == "\r":
            position += 1
        if position >= size or text[position] != "\n":
            break
        line = position + 1
        if _indent_end(text, line) - line <= key_indent:
            break
        match = _CONTINUATION_LINE.match(text, line)
        if match is None:
            break
        end = match.end("literal")
        position = match.end()
        if match["closer"]:
            break
    return end


def _escaped_chunk(text: str, position: int, extra: int | None = None) -> tuple[int, int, bool] | None:
    """(string end, chunk end, ends the value) of the escaped-newline chunk at position; extra None picks a level."""
    head = _ESCAPED_CHUNK_HEAD.match(text, position)
    if head is None:
        return None
    start = head.end()
    quote = start + 1 if text[start] == "b" and text[start + 1] == "\\" else start
    if text[quote] == "\\":
        if extra is None:
            literal_end, closed, _ = _chosen_end(text, quote)
        else:
            literal_end, closed, _ = _string_end(text, quote, extra)
    else:
        literal = _STRING_ONLY.match(text, start)
        literal_end, closed = (literal.end(), True) if literal is not None else (start, False)
    tail = _ESCAPED_CHUNK_TAIL.match(text, literal_end) if closed else None
    if tail is None:
        return None
    return literal_end, tail.end(), bool(tail["closer"])


def _escaped_continuation_end(text: str, end: int, extra: int | None = None) -> int:
    """Extend a closed string value over following repr-escaped lines made of one string literal each."""
    position = end
    while (chunk := _escaped_chunk(text, position, extra)) is not None:
        end, position = chunk[0], chunk[1]
        if chunk[2]:
            break
    return end


def _quoted_key_spans(text: str) -> Iterator[tuple[int, int, re.Match[str]]]:
    """Matches of the quoted-key pattern with the span extended over a multi-line value.
    An escaped key matches only its container opener; the end scan finds the rest."""
    pos = 0
    line = 0
    scanned = 0
    while (match := next(_QUOTED_KEY_PATTERN.finditer(text, pos), None)) is not None:
        start, end = match.span()
        value = match.span("value")[0]
        container = _CONTAINER_HEAD.match(text, value)
        if match["vq"]:
            end, closed, extra = _chosen_end(text, value + (text[value] == "b"))
            if closed:
                end = _escaped_continuation_end(text, end, extra)
            yield start, end, match
            pos = end
            continue
        closed_string = container is None and _STRING_ONLY.fullmatch(text, value, end) is not None
        if container is None and not closed_string:
            # A value running to the line end must not swallow a trailing comma: the next key needs its lead.
            while end > value + 1 and text[end - 1] in " \t,":
                end -= 1
        if closed_string and _escaped_chunk(text, end) is not None:
            end = _escaped_continuation_end(text, end)
        elif container is not None or (closed_string and text.startswith(("\n", "\r\n"), end)):
            quote = match.span("quote")[0]
            newline = text.rfind("\n", scanned, quote)
            if newline >= 0:
                line = newline + 1
            scanned = quote
            key_indent = _indent_end(text, line) - line
            if container is not None:
                scan_end, deeper = _container_end(text, container.end() - 1, key_indent)
                if len(match["quote"]) > 1 or deeper:
                    end = scan_end
                elif scan_end < end and not text[scan_end:end].strip(" \t,\r"):
                    end = scan_end
                else:
                    end = max(end, scan_end)
            else:
                end = _string_continuation_end(text, end, key_indent)
        yield start, end, match
        pos = end


def _sub_quoted_keys(text: str) -> str:
    out = io.StringIO()
    cursor = 0
    for start, end, match in _quoted_key_spans(text):
        out.write(text[cursor:start])
        out.write(match.expand(_QUOTED_KEY_REPL))
        cursor = end
    out.write(text[cursor:])
    return out.getvalue()


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
    r"(?<![\w.$#/-])(?!\d+/)(?P<user>[\w.$#-]+)/[^\s/]+@"
    r"(?=//|\(|[\w.-]+:\d+(?!\w)|(?!\d+(?:\.\d+){1,2}/|\d+\.\d+\.\d+[-+][\w.]*/)[\w.-]+/\w)"
)
# Slash in userinfo right after `://`; the dotless user keeps `unpkg.com/react@18/` and `github.com/user@host` intact.
_SCHEME_SLASH_DSN_PATTERN = re.compile(r"(?<=://)(?P<user>[\w$#~-]+)/[^\s/@]+@(?=[\w.-]+(?:[/:?#\s)\]'\"]|$))")

_PLAIN_URI_SCHEMES = (
    "http|https|s3|s3a|s3n|gs|gcs|abfs|abfss|wasb|wasbs|az|adl|file|ftp|sftp|hdfs|"
    "postgres|postgresql|mysql|mongodb|redis|sqlite|snowflake|mssql|ws|wss|ssh|git|grpc|kafka|ftps|"
    "oracle|redshift|bigquery|databricks|" + _DB_SCHEMES + "|duckdb"
)
# `user:password://host`: the password sits in the scheme position. Delimiters are excluded from both classes
# and the lookbehind anchors the start, so the scan stays linear. The password may hold `@` and be followed by one
# `:word`; a `/` may precede the user. `jdbc:` is never the user or the head of the password.
_DELIM = "\\s\"'(\\[{<=,;"
_LOG_LEVELS = "DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL|TRACE|NOTICE"
_PLAIN_SCHEME_GUARD = rf"(?!(?i:{_PLAIN_URI_SCHEMES})(?:\+[\w.-]+)?://)"
_PASSWORD_AS_SCHEME_PATTERN = re.compile(
    rf"(?<![^{_DELIM}/])(?!(?i:jdbc):)"
    # A log level before an extra word (`ERROR:Failed:x://`) is a prefix, not a user.
    rf"(?!(?i:{_LOG_LEVELS}):[^{_DELIM}:/]+:{_PLAIN_SCHEME_GUARD}[^{_DELIM}:/@]+://)"
    rf"(?P<user>[^{_DELIM}:/@]+):"
    rf"(?!(?i:jdbc):[^/]){_PLAIN_SCHEME_GUARD}(?P<pw>[^{_DELIM}:/]+(?::{_PLAIN_SCHEME_GUARD}[^{_DELIM}:/@]+)?)(?=://)"
)


# The jdbc tail (used by _uri_projection and the jdbc marker cut): non-whitespace text after a jdbc match, cut at the
# next `://` so repeated URIs scan linearly.
_JDBC_TAIL_PATTERN = re.compile(r"(?:(?!://)\S)*")
_KEY_VALUE_REGION_PATTERN = re.compile(r":[^=;:]*=")
# Private-use marker appended by _scrub_uri_match after a dropped query, key=value path or hidden percent path;
# _cut_marker_tail cuts the token tail that follows it.
_JDBC_CUT = "\ue001"
# The cut runs through the token, stopping only before a real chain `::` or a `,`/`;` starting a new `scheme://` item.
# A non-chain `::` swallows its whole protocol run, so the bounded run fails closed instead of stopping mid-run.
_CHAIN_AFTER_VALUE_PATTERN = re.compile(_CHAIN_AFTER_VALUE)
_JDBC_CUT_PATTERN = re.compile(
    _JDBC_CUT + rf"(?:(?!{_CHAIN_AFTER_VALUE}|[,;][\"'(\[{{<]*[A-Za-z][A-Za-z0-9+.-]*://)"
    r"(?:::(?:[a-z][a-z0-9+.-]{0,31}::)*+|\S))*"
)
# Own marker for a dropped jdbc path: its cut runs to the next `://` (jdbc tails are never fsspec chains), then
# through spaced `key = value` and quoted or braced values.
_JDBC_PATH_CUT = "\ue002"
# A quoted value closes only before a delimiter (a dict repr's quote stays a boundary); values stop at a newline or `://`.
_JDBC_VALUE_END = r"""[;\s,)\]}>\\]|$|[.!?:]+(?:\s|$|["'])"""
_JDBC_GLUED = r"""(?:(?!://)[^;\s'"])++"""
_JDBC_QUOTED = (
    rf"""'(?:(?!://)[^'\n]|'')*+(?:'(?={_JDBC_VALUE_END}|["'](?:{_JDBC_VALUE_END}))|'{_JDBC_GLUED}(?!')|(?=\n|$|://))"""
    rf"""|"(?:(?!://)[^"\n]|"")*+(?:"(?={_JDBC_VALUE_END}|["'](?:{_JDBC_VALUE_END}))|"{_JDBC_GLUED}(?!")|(?=\n|$|://))"""
)
_JDBC_BRACED = r"\{(?:(?!://)[^}\n]|\}\})*+\}?"
# Own marker for a hidden parameter region after a mid-token stop: the cut keeps the tail's clean head.
_HIDDEN_CUT = "\ue003"
_MARKER_PATTERN = re.compile(f"[{_JDBC_CUT}{_JDBC_PATH_CUT}{_HIDDEN_CUT}]")


def _value_cut_pattern(marker: str) -> re.Pattern[str]:
    return re.compile(
        marker + rf"(?:(?P<v>(?<=[= \t])(?:{_JDBC_QUOTED}|{_JDBC_BRACED})|(?<={marker}){_JDBC_BRACED})"
        rf"|[ \t]++(?==)|(?<==)[ \t]++|(?<=[= \t]){_KNOWN_SCHEMES}[ \t]++"
        r"|(?!://|[,;][\"'(\[{<]*(?:(?i:jdbc):)?[A-Za-z][A-Za-z0-9+.-]*://)\S)*"  # not `*+`: 3.11 misreports group spans
    )


_JDBC_PATH_CUT_PATTERN = _value_cut_pattern(_JDBC_PATH_CUT)
_HIDDEN_CUT_PATTERN = _value_cut_pattern(_HIDDEN_CUT)
# Closers then sentence punctuation then whitespace or end: the match ended a value, nothing was cut off.
_CLOSING_END_PATTERN = re.compile(r"[\"'<>)\]}]+[,:;.!?]*(?:\s|$)")
# Closers then a separator and an opener: a following list or dict item, if the URL itself was opened.
_CLOSING_NEXT_PATTERN = re.compile(r"[\"'<>)\]}]+[,:][\"'\[{(<]")
_OPENERS = "\"'([{<"
_URI_LIST_ITEM_PATTERN = re.compile(r"[,;][\"'(\[{<]*(?:(?i:jdbc):)?[A-Za-z][A-Za-z0-9+.-]*://")
# Closers kept after a cut so a later `, 'password': ...` still has its `,` lead.
_CLOSERS = "\"'<>)]},"
# Characters the free-text URI pattern refuses inside a query or fragment (a path refuses all but `)` too):
# a match ending right before one of them may have stopped in the middle of a dropped value.
_MATCH_END_TERMINATORS = frozenset("\"'<>)]")
# The opener a terminator closes when the whole URL was quoted or bracketed.
_OPENER_OF = {'"': '"', "'": "'", ">": "<", ")": "(", "]": "["}


def _cut_key_value_path(path: str) -> str:
    """The path up to (excluding) the first segment that looks like key=value after a ':'."""
    segments = path.split("/")
    for index, segment in enumerate(segments):
        if "=" in segment.partition(":")[2]:
            return "/".join(segments[:index])
    return path


def _azure_container_prefix(match: re.Match[str]) -> str:
    """`container@` when the URI names an Azure container, else empty."""
    userinfo = match["userinfo"]
    if (
        userinfo
        and match["scheme"].lower() in _AZURE_CONTAINER_SCHEMES
        and _AZURE_CONTAINER_PATTERN.fullmatch(userinfo)
    ):
        return f"{userinfo}@"
    return ""


_SPACED_EQUALS_PATTERN = re.compile(r"[ \t]+=(?!=)")
_KNOWN_SCHEME_END_PATTERN = re.compile(rf"={_KNOWN_SCHEMES}\Z")


def _opens_brace_value(text: str) -> bool:
    """True when the text ends inside a `={...}` value (`}}` escapes a brace)."""
    inside = False
    i = 0
    while i < len(text):
        if inside:
            if text.startswith("}}", i):
                i += 2
                continue
            inside = text[i] != "}"
        elif text.startswith("={", i):
            inside = True
            i += 1
        i += 1
    return inside


def _in_open_brace_value(text: str, start: int) -> bool:
    """True when the nearby text before start sits inside an open `={...}` value."""
    opener = text.rfind("={", max(0, start - 256), start)
    return opener >= 0 and _opens_brace_value(text[opener:start])


def _cut_resume(text: str) -> str:
    """What stays after the marker so the cut resumes inside an open value."""
    if _opens_brace_value(text):
        return "{"
    return "=" if text.endswith("=") or _KNOWN_SCHEME_END_PATTERN.search(text) else ""


def _dropped_continues(match: re.Match[str], dropped: str) -> bool:
    """True when dropped URI text hands its value on past the match (`=` end, open brace, spaced `=`, `=Bearer `)."""
    if not dropped:
        return False
    if dropped.endswith("=") or _opens_brace_value(dropped):
        return True
    text, end = match.string, match.end()
    if _KNOWN_SCHEME_END_PATTERN.search(dropped):
        return text[end : end + 1] in (" ", "\t")
    last_parameter = dropped[max(dropped.rfind(mark) for mark in "?&;#") + 1 :]
    if "=" in last_parameter and not match["scheme"].lower().startswith("jdbc:"):
        return False
    return _SPACED_EQUALS_PATTERN.match(text, end) is not None


def _uri_projection(match: re.Match[str]) -> str | None:
    scheme, host = match["scheme"], match["host"]
    # Host parameters ride along with the path: the same key=value and hidden-percent checks apply.
    path = (match["params"] or "") + (match["path"] or "")
    container = _azure_container_prefix(match)
    is_jdbc = scheme.lower().startswith("jdbc:")
    if is_jdbc:
        tail = _JDBC_TAIL_PATTERN.search(match.string, match.end())
        if any(mark in path + (tail[0] if tail else "") for mark in "=;@"):
            return f"{scheme}://{container}{host}{_JDBC_PATH_CUT}{_cut_resume(path)}"
    key_value_path = any("=" in segment.partition(":")[2] for segment in path.split("/"))
    if key_value_path and not is_jdbc:
        return None
    query = match["query"] or ""
    if _dropped_continues(match, query):
        kept = _cut_hidden_percent(scheme, host, path)
        return f"{scheme}://{container}{host}{kept}{_JDBC_PATH_CUT}{_cut_resume(query)}"
    if key_value_path:
        return None
    return f"{scheme}://{container}{host}{_cut_hidden_percent(scheme, host, path)}"


def _cut_hidden_percent(scheme: str, host: str, path: str) -> str:
    """Keep a percent-encoded path unless its decoded form hides a credential; then cut at the first escape."""
    if "%" not in path:
        return path
    decoded = unquote(path)
    probe = f"{scheme}://{host}{decoded}"
    hides = "%" in decoded or any(mark in decoded for mark in ("@", "'", '"', "://", *_REGISTERED_ESCAPED.get()))
    if not hides and _scrub_passes(probe) == probe:
        return path
    head = path.partition("%")[0]
    return head[: head.rfind("/") + 1]


def _part_clean_head(part: str) -> str:
    """The part up to its first raw `?`/`#`, else its clean head before a `;` or percent-escape."""
    raw = min((i for i in (part.find("?"), part.find("#")) if i >= 0), default=len(part))
    decoded = _decode_to_fixpoint(part[:raw])
    if decoded is not None and not any(mark in decoded for mark in ";?#"):
        return part[:raw]
    return _clean_head(part)


def _token_tail(text: str, start: int) -> str:
    found = _JDBC_TAIL_PATTERN.match(text, start)
    return found[0] if found else ""


def _hides_parameter_region(match: re.Match[str]) -> bool:
    """True when the token tail after a mid-token stop char hides a parameter region.
    A closer the URL was opened with ends the value; a `[,;]` starting another `scheme://` item is a list."""
    text, end, start = match.string, match.end(), match.start()
    if end >= len(text) or text[end].isspace():
        return False
    if text[end] in _MATCH_END_TERMINATORS:
        if start > 0 and text[start - 1] == _OPENER_OF.get(text[end]):
            # A closer glued to a token character is mid-path, not the end of the opened value.
            following = text[end + 1 : end + 2]
            if not (following.isalnum() or following == "_"):
                return False
    # A `}` stop inside an open braced value ends it; the jdbc cut owns that value.
    elif (text[end] == "}" and _in_open_brace_value(text, start)) or _URI_LIST_ITEM_PATTERN.match(text, end):
        return False
    decoded = _decode_to_fixpoint(_token_tail(text, end))
    return decoded is None or bool(_KEY_VALUE_REGION_PATTERN.search(decoded) or _PARAMETER_REGION.search(decoded))


def _hidden_head(text: str, start: int) -> str:
    """The clean head of the token tail at start, up to its first parameter region."""
    if text[start] in _MATCH_END_TERMINATORS:
        return ""
    tail = _token_tail(text, start)
    regions = _parameter_regions(tail)
    if not regions:
        return ""
    first, _, end = regions[0]
    return _cut_key_value_path(tail[:first] + _part_clean_head(tail[first:end]))


def _cut_hidden_tail(marker: re.Match[str]) -> str:
    return _hidden_head(marker.string, marker.start() + 1) + _cut_path_marker_tail(marker)


def _cut_dropped_tail(match: re.Match[str], scrubbed: str, dropped: bool) -> str:
    """Append the cut marker when a dropped value may continue past the match (it stops at a quote or bracket)."""
    if scrubbed.rstrip("={").endswith((_JDBC_CUT, _JDBC_PATH_CUT)):
        return scrubbed
    if not dropped:
        return scrubbed + _HIDDEN_CUT if _hides_parameter_region(match) else scrubbed
    text, end, start = match.string, match.end(), match.start()
    if text.startswith("::", end) and _CHAIN_AFTER_VALUE_PATTERN.match(text, end) is None:
        return scrubbed + _JDBC_CUT
    if end >= len(text) or text[end] not in _MATCH_END_TERMINATORS:
        return scrubbed
    if _CLOSING_END_PATTERN.match(text, end):
        return scrubbed
    if _CLOSING_NEXT_PATTERN.match(text, end) and start > 0 and text[start - 1] in _OPENERS:
        return scrubbed
    return scrubbed + _JDBC_CUT


def _cut_marker_tail(marker: re.Match[str]) -> str:
    """Drop the token tail after the marker, keeping only its trailing run of closers."""
    tail = marker[0][1:]
    return tail[len(tail.rstrip(_CLOSERS)) :]


_PUNCTUATED_CLOSERS = re.compile(r"[\"'<>)\]},]*[,:;.!?]*")


def _cut_path_marker_tail(marker: re.Match[str]) -> str:
    """Like _cut_marker_tail, but only the text after the last quoted or braced value can hold closers."""
    start = marker.end("v") if marker["v"] is not None else marker.start() + 1
    tail = marker.string[start : marker.end()]
    if marker["v"] is not None and _PUNCTUATED_CLOSERS.fullmatch(tail):
        return tail
    return tail[len(tail.rstrip(_CLOSERS)) :]


def _keyword_run(text: str, pos: int, end: int, limit: int) -> int | None:
    """End of the `;k=v` / `&k=v` run from pos reaching end (or the whitespace-terminated URI end) with each element masked; None if not."""
    while pos < len(text) and text[pos] in ";&":
        found = _KEYWORD_PATTERN.match(text, pos + 1)
        if found is None:
            return None
        pos = found.end()
        if (pos >= limit and text[limit : limit + 1].isspace()) or (
            pos >= end and text[pos : pos + 1] not in (";", "&")
        ):
            return pos
    return None


# Up to three following tokens (each capped, so the window is constant work): enough to hold a `= value` or `Bearer value` that follows a URI match.
_STRADDLE_WINDOW = re.compile(r"(?:[ \t]*\S{1,64}){0,3}")


def _straddling_keyword(match: re.Match[str], start: int) -> str:
    """`;` plus the keyword head of a keyword pass match that starts in the dropped span but ends past the URI."""
    text, limit = match.string, match.end()
    window = _STRADDLE_WINDOW.match(text, limit)
    stop = window.end() if window else limit
    for kw in _KEYWORD_PATTERN.finditer(text, start, stop):
        if kw.start() >= limit:
            break
        if kw.end() > limit:
            return ";" + text[kw.start() : limit]
    return ""


def _cut_at_parameter_region(match: re.Match[str], path: str) -> str | None:
    """The URI cut back to the first parameter region the keyword pass will not mask."""
    base = match.start("params") if match["params"] else match.start("path")
    text = match.string
    validated = 0
    for start, head, end in _parameter_regions(path):
        if base + head < validated:
            continue
        run_end = _keyword_run(text, base + head, base + end, match.end())
        if run_end is not None:
            validated = run_end
            continue
        if not any(c.isalnum() or c == "_" for c in path[head:end]):
            continue
        scheme, host = match["scheme"], match["host"]
        cut = _cut_key_value_path(path[:start])
        kept = _azure_container_prefix(match)
        return f"{scheme}://{kept}{host}{_cut_hidden_percent(scheme, host, cut)}" + _straddling_keyword(
            match, base + start
        )
    return None


def _scrub_uri_match(match: re.Match[str]) -> str:
    scheme, host = match["scheme"], match["host"]
    path = (match["params"] or "") + (match["path"] or "")
    if path and not scheme.lower().startswith("jdbc:"):
        cut_uri = _cut_at_parameter_region(match, path)
        if cut_uri is not None:
            dropped = path + (match["query"] or "")
            if _dropped_continues(match, dropped):
                cut_uri += _JDBC_PATH_CUT + _cut_resume(dropped)
            return _cut_dropped_tail(match, cut_uri, True)
    projection = _uri_projection(match)
    if projection is not None:
        dropped = bool(match["query"]) or not projection.endswith(path)
        return _cut_dropped_tail(match, projection, dropped)
    scrubbed = f"{scheme}://{host}{_cut_hidden_percent(scheme, host, _cut_key_value_path(path))}"
    query = match["query"] or ""
    if _dropped_continues(match, query):
        scrubbed += _JDBC_PATH_CUT + _cut_resume(query)
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


def _mask_path_password(match: re.Match[str]) -> str:
    """A free-text userinfo match left after strip_uri_userinfo: mask the path piece, keep the authority."""
    scheme = match["scheme"]
    user, _, rest = match[0][len(scheme) + 3 :].partition(":")
    digits = rest.partition("/")[0]
    if not digits.isascii() or not digits.isdigit():
        return f"{scheme}://{user}:***@"
    # Digits are a port except on object-store schemes, where the path piece is a password.
    if scheme.lower() not in _OBJECT_STORE_SCHEMES:
        return match[0]
    return f"{scheme}://{user}:{digits}/***@"


def mask_identity_credentials(text: str) -> str:
    """Mask, keeping the authority, a password read from the path after `host:port/` and per-segment keyword values."""
    return _KEYWORD_SEGMENT_PATTERN.sub(_keyword_repl, _FREE_TEXT_USERINFO_PATTERN.sub(_mask_path_password, text))


_MAX_DECODE_ROUNDS = 4
# `::` separates parts only where an fsspec chain continues: before a new `<scheme>://`
# element or a further lowercase `<protocol>::` element. Any other `::` (for example inside
# a path parameter value) stays inside its part, so the region cut below removes it too.
_PART_SPLIT = re.compile(rf"(/|{_CHAIN_COLONS})")
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


def _parameter_regions(text: str) -> list[tuple[int, int, int]]:
    """(part start, clean-head end, part end) per part holding a `;?#` region."""
    regions: list[tuple[int, int, int]] = []
    pos = 0
    for i, part in enumerate(_PART_SPLIT.split(text)):
        if i % 2 == 0:
            decoded = _decode_to_fixpoint(part)
            if decoded is None or _PARAMETER_REGION.search(decoded):
                regions.append((pos, pos + len(_clean_head(part)), pos + len(part)))
        pos += len(part)
    return regions


def _cut_path_parameter_regions(uri: str) -> str:
    """Cut every parameter-region part to its clean head + `;***`."""
    out: list[str] = []
    cursor = 0
    for start, head, end in _parameter_regions(uri):
        out.append(uri[cursor:head] + ";***")
        cursor = end
    out.append(uri[cursor:])
    return "".join(out)


_REGISTERED_SECRETS: ContextVar[frozenset[str]] = ContextVar("mloda_registered_secrets", default=frozenset())
_IDENTITY_SECRETS: ContextVar[frozenset[str]] = ContextVar("mloda_identity_secrets", default=frozenset())
_REGISTERED_ESCAPED: ContextVar[frozenset[str]] = ContextVar("mloda_registered_escaped", default=frozenset())


def _with_escaped(literals: frozenset[str]) -> frozenset[str]:
    """The literals plus their repr (both quote styles), JSON and non-ASCII JSON escaped forms."""
    forms = set(literals)
    for literal in literals:
        if literal:
            forms.add(repr(literal)[1:-1])
            forms.add(repr(literal + '"')[1:-2])
            forms.add(json.dumps(literal)[1:-1])
            forms.add(json.dumps(literal, ensure_ascii=False)[1:-1])
    return frozenset(forms)


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

    Patterns run on the original text so a literal never deletes their anchors. Origins are exact for the URI
    pass and a greedy subsequence alignment elsewhere; output chars that align to a registered literal are masked.
    """
    registered = _REGISTERED_ESCAPED.get()
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


def _uri_origins(match: re.Match[str], replacement: str, origin: list[int]) -> list[int]:
    """Exact origins for a URI-pass replacement: scheme, kept Azure container and host onward come from group spans."""
    base = match.start()
    lead = len(match["scheme"]) + 3
    result = [origin[match.start("scheme") - base + i] for i in range(lead)]
    userinfo = match["userinfo"]
    rest = lead
    if userinfo and replacement.startswith(f"{userinfo}@", lead):
        span = match.start("userinfo") - base
        result.extend(origin[span + i] for i in range(len(userinfo) + 1))
        rest += len(userinfo) + 1
    host = match.start("host") - base
    size = match.end() - base
    text = match.string
    at = match.start() + host
    result.extend(
        origin[host + i] if host + i < size and text[at + i] == replacement[rest + i] else -1
        for i in range(len(replacement) - rest)
    )
    return result


def _sub_tracked(
    pattern: re.Pattern[str], repl: Any, text: str, origin: list[int], origin_of: Any = None
) -> tuple[str, list[int]]:
    out = io.StringIO()
    origins: list[int] = []
    cursor = 0
    spans = (
        _quoted_key_spans(text) if pattern is _QUOTED_KEY_PATTERN else ((*m.span(), m) for m in pattern.finditer(text))
    )
    for start, end, match in spans:
        out.write(text[cursor:start])
        origins.extend(origin[cursor:start])
        replacement = _replacement(match, repl)
        out.write(replacement)
        if origin_of is None:
            origins.extend(_align(text[start:end], origin[start:end], replacement))
        else:
            origins.extend(origin_of(match, replacement, origin[start:end]))
        cursor = end
    out.write(text[cursor:])
    origins.extend(origin[cursor:])
    return out.getvalue(), origins


def _authorization_repl(m: re.Match[str]) -> str:
    return m["head"] + (f"{m['scheme']} ***" if m["scheme"] else "***")


def _keyword_repl(m: re.Match[str]) -> str:
    keyword, sep, value = m["keyword"], m["sep"], m["value"]
    text, start, end = m.string, m.start(), m.end()
    if (
        keyword in ("PWD", "OLDPWD")
        and not m["known"]
        and (start == 0 or text[start - 1].isspace())
        and _PATH_VALUE.fullmatch(value)
        and value.split("/")[1] in _PWD_ROOTS
        and (end == len(text) or text[end].isspace())
    ):
        return m[0]
    if _COUNT_NAME.fullmatch(keyword.rpartition(".")[2]) and value.isascii() and value.isdigit() and len(value) <= 7:
        return m[0]
    closed = m["braced"] or (m["quoted"] and value[-1] == value[0])
    if closed and end < len(text) and text[end] not in ";&'\"" and not text[end].isspace():
        closer = "}" if m["braced"] else value[0]
        opener = "{" if m["braced"] else value[0]
        return f"{keyword}{sep}{opener}***{closer}"
    return f"{keyword}{sep}***"


_SOURCE_EXTENSION = re.compile(
    r"\.(?:py|pyx|ipynb|js|ts|java|scala|kt|go|rs|rb|sql|yaml|yml|json|toml|cfg|ini|txt|csv|log|sh)$", re.IGNORECASE
)


_SCHEME_TOKEN = re.compile(r"[a-z][a-z0-9+.-]*")


def _password_scheme_repl(m: re.Match[str]) -> str:
    """Keep a `file.py:12:` location prefix; mask any other `user:password` before a scheme."""
    user = m["user"]
    line, _, tail = m["pw"].partition(":")
    if _SOURCE_EXTENSION.search(user) and line.isascii() and line.isdigit() and _SCHEME_TOKEN.fullmatch(tail):
        return m[0]
    return f"{user}:***"


# Must run after the URI pass, which leaves secret-named `;k=v` regions to it.
_PASSES: list[tuple[re.Pattern[str], Any]] = [
    (_DB_USERINFO_PATTERN, r"\g<scheme>://"),
    (_FREE_TEXT_USERINFO_PATTERN, r"\g<scheme>://"),
    (_FREE_TEXT_URI_PATTERN, _scrub_uri_match),
    (_JDBC_CUT_PATTERN, _cut_marker_tail),
    (_JDBC_PATH_CUT_PATTERN, _cut_path_marker_tail),
    (_HIDDEN_CUT_PATTERN, _cut_hidden_tail),
    (_PASSWORD_AS_SCHEME_PATTERN, _password_scheme_repl),
    (_AUTHORIZATION_PATTERN, _authorization_repl),
    (_BEARER_PATTERN, r"\g<scheme> ***"),
    (_KEYWORD_PATTERN, _keyword_repl),
    (_QUOTED_KEY_PATTERN, _QUOTED_KEY_REPL),
    (_SLASH_DSN_PATTERN, r"\g<user>/***@"),
    (_SCHEME_SLASH_DSN_PATTERN, r"\g<user>/***@"),
]


_TRAILING = " \t\r"
_UNESCAPED = {"\\": "\\", "'": "'", '"': '"', "n": "\n", "t": "\t", "r": "\r"}


def _tail_literal(text: str, begin: int, end: int) -> re.Match[str] | None:
    """The literal ending the line text[begin:end], found backward so an earlier stray quote cannot shift it."""
    while end > begin and text[end - 1] in _TRAILING:
        end -= 1
    if end - begin < 2 or text[end - 1] not in "'\"":
        return None
    quote, i = text[end - 1], end - 2
    while i >= begin:
        if text[i] != quote:
            i -= 1
            continue
        j = i
        while j > begin and text[j - 1] == "\\":
            j -= 1
        if (i - j) % 2 == 0:
            head = i - 1 if i > begin and text[i - 1] == "b" else i
            return _STRING_ONLY.fullmatch(text, head, end) or _STRING_ONLY.fullmatch(text, i, end)
        i = j - 1
    return None


def _literal_runs(text: str) -> Iterator[list[re.Match[str]]]:
    """Runs of two or more string literals on consecutive lines; each line is scanned once."""
    run: list[re.Match[str]] = []
    begin = 0
    for line in text.split("\n"):
        end = begin + len(line)
        first = begin
        while first < end and text[first] in " \t":
            first += 1
        chunk = _STRING_ONLY.match(text, first, end) if run else None
        if chunk is not None:
            run.append(chunk)
            if not text[chunk.span()[1] : end].strip(_TRAILING):
                begin = end + 1
                continue
        if len(run) > 1:
            yield run
        tail = _tail_literal(text, begin, end)
        run = [tail] if tail else []
        begin = end + 1
    if len(run) > 1:
        yield run


def _wrapped_spans(text: str) -> list[tuple[int, int]]:
    """Source spans of the secrets in each run of adjacent string literals, found on the run's joined value."""
    spans: list[tuple[int, int]] = []
    for run in _literal_runs(text):
        value = io.StringIO()
        where: list[tuple[int, int, int]] = []
        for chunk, literal in enumerate(run):
            first, last_end = literal.span()
            i, stop = first + (text[first] == "b") + 1, last_end - 1
            while i < stop:
                width = 2 if text[i] == "\\" and text[i + 1] in _UNESCAPED else 1
                value.write(_UNESCAPED[text[i + 1]] if width == 2 else text[i])
                where.append((i, i + width, chunk))
                i += width
        joined = value.getvalue()
        # The joined value is scrubbed without this pass, so the recursion through _cut_hidden_percent ends.
        origin = _run_passes(joined, list(range(len(joined))))[1]
        masked = [True] * len(joined)
        for source in origin:
            if source >= 0:
                masked[source] = False
        for start, end in _literal_spans(joined, _REGISTERED_ESCAPED.get()):
            masked[start:end] = [True] * (end - start)
        last: tuple[int, int, int] | None = None
        for index, flag in enumerate(masked):
            if not flag:
                last = None
            elif last is not None and last[2] == where[index][2]:
                last = (last[0], where[index][1], last[2])
                spans[-1] = last[:2]
            else:
                last = where[index]
                spans.append(last[:2])
    return spans


def _mask_wrapped(text: str) -> tuple[str, list[int]]:
    out = io.StringIO()
    origin: list[int] = []
    cursor = 0
    for start, end in _wrapped_spans(text):
        out.write(text[cursor:start] + "***")
        origin.extend(range(cursor, start))
        origin.extend((-1, -1, -1))
        cursor = end
    out.write(text[cursor:])
    origin.extend(range(cursor, len(text)))
    return out.getvalue(), origin


def _scrub_passes(text: str) -> str:
    text = _mask_wrapped(_MARKER_PATTERN.sub("\ufffd", text))[0]
    for pattern, repl in _PASSES:
        text = _sub_quoted_keys(text) if pattern is _QUOTED_KEY_PATTERN else pattern.sub(repl, text)
    return text


def _run_passes(text: str, origin: list[int]) -> tuple[str, list[int]]:
    for pattern, repl in _PASSES:
        text, origin = _sub_tracked(
            pattern, repl, text, origin, _uri_origins if pattern is _FREE_TEXT_URI_PATTERN else None
        )
    return text, origin


def _scrub_tracked(text: str) -> tuple[str, list[int]]:
    return _run_passes(*_mask_wrapped(_MARKER_PATTERN.sub("\ufffd", text)))


# Bounds recursion so a deeply nested attribute cannot hit RecursionError on the failure path.
_MAX_MASK_DEPTH = 32


def _mask_item(item: Any, literals: frozenset[str], memo: dict[int, Any] | None = None, depth: int = 0) -> Any:
    """Mask str leaves in tuples, lists, dicts, sets and frozensets; return ``item`` itself when nothing changed."""
    if isinstance(item, str):
        masked_str = mask_literals(item, literals)
        return item if masked_str == item else masked_str
    if not isinstance(item, (tuple, list, dict, set, frozenset)):
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
    if isinstance(item, (set, frozenset)):
        memo[id(item)] = "***"
        members = [_mask_item(element, literals, memo, depth + 1) for element in item]
        if all(new is old for new, old in zip(members, item)):
            memo[id(item)] = item
            return item
        rebuilt = set(members) if isinstance(item, set) else frozenset(members)
        memo[id(item)] = rebuilt
        return rebuilt
    memo[id(item)] = "***"
    elements = [_mask_item(element, literals, memo, depth + 1) for element in item]
    if all(new is old for new, old in zip(elements, item)):
        memo[id(item)] = item
        return item
    result = item._make(elements) if hasattr(item, "_make") else tuple(elements)
    memo[id(item)] = result
    return result


_STR_FIELDS: tuple[tuple[type[BaseException], tuple[str, ...]], ...] = (
    (OSError, ("strerror", "filename", "filename2")),
    (ImportError, ("msg", "name", "path")),
    (SyntaxError, ("msg", "filename", "text")),
    (UnicodeEncodeError, ("object", "reason")),
    (UnicodeDecodeError, ("reason",)),
    (UnicodeTranslateError, ("object", "reason")),
    (SystemExit, ("code",)),
)


def redact_exception(exc: BaseException, literals: frozenset[str]) -> BaseException:
    """Mask in place and return the exception, or a copy when a group's read-only message had to change."""
    if not literals:
        return exc
    return _redact_forms(exc, _with_escaped(literals))


def _redact_forms(exc: BaseException, literals: frozenset[str]) -> BaseException:
    """Mask the already expanded ``literals`` in place across the exception graph."""
    seen: set[int] = set()
    visited: list[BaseException] = []
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        visited.append(current)
        masked_args = _mask_item(current.args, literals)
        if masked_args is not current.args:
            object.__setattr__(current, "args", masked_args)
        attributes = vars(current)
        for name, value in list(attributes.items()):
            if isinstance(value, (str, dict, list, tuple, set, frozenset)) and not name.startswith("__"):
                masked = _mask_item(value, literals)
                if masked is not value:
                    attributes[name] = masked
        for base, names in _STR_FIELDS:
            if not isinstance(current, base):
                continue
            for name in names:
                # The base's own descriptor, written only on change: a None in these fields alters OSError.__str__.
                descriptor = getattr(base, name)
                value = descriptor.__get__(current)
                if isinstance(value, str):
                    masked_value = mask_literals(value, literals)
                    if masked_value != value:
                        descriptor.__set__(current, masked_value)
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
    groups = [g for g in visited if isinstance(g, BaseExceptionGroup)]
    if not any(mask_literals(g.message, literals) != g.message for g in groups):
        return exc
    memo: dict[int, BaseException] = {}
    for group in groups:
        _rebuilt_group(group, literals, memo, 0)
    _rewire([*visited, *memo.values()], memo)
    return memo.get(id(exc), exc)


def _rebuilt_group(
    group: BaseExceptionGroup[Any], forms: frozenset[str], memo: dict[int, BaseException], depth: int
) -> BaseException:
    """Return ``group`` or a copy with the forms masked in its message and nested groups."""
    if id(group) in memo:
        return memo[id(group)]
    if depth > _MAX_MASK_DEPTH:
        memo[id(group)] = RuntimeError("***") if isinstance(group, Exception) else BaseException("***")
        return memo[id(group)]
    original = group.exceptions
    members = [_rebuilt_group(m, forms, memo, depth + 1) if isinstance(m, BaseExceptionGroup) else m for m in original]
    members_changed = any(new is not old for new, old in zip(members, original))
    message = mask_literals(group.message, forms)
    if message == group.message and not members_changed:
        memo[id(group)] = group
        return group
    copy = BaseExceptionGroup.__new__(type(group), message, members)
    copy.__dict__.update(group.__dict__)
    args = list(group.args)
    for index, element in enumerate(args):
        if (
            isinstance(element, (list, tuple))
            and len(element) == len(original)
            and all(a is b for a, b in zip(element, original))
        ):
            args[index] = list(members)
    object.__setattr__(copy, "args", tuple(args))
    copy.with_traceback(group.__traceback__)
    object.__setattr__(copy, "__cause__", group.__cause__)
    object.__setattr__(copy, "__context__", group.__context__)
    object.__setattr__(copy, "__suppress_context__", group.__suppress_context__)
    memo[id(group)] = copy
    return copy


def _rewire(nodes: list[BaseException], memo: dict[int, BaseException]) -> None:
    """Point causes, contexts and member containers at the rebuilt groups."""
    for current in nodes:
        for name in ("__cause__", "__context__"):
            target = getattr(current, name)
            replacement = memo.get(id(target)) if target is not None else None
            if replacement is not None and replacement is not target:
                suppress = current.__suppress_context__
                object.__setattr__(current, name, replacement)
                object.__setattr__(current, "__suppress_context__", suppress)
        attributes = vars(current)
        for name in ("refusals", "exceptions"):
            value = attributes.get(name)
            if isinstance(value, (tuple, list)):
                swapped = [memo.get(id(item), item) for item in value]
                if any(new is not old for new, old in zip(swapped, value)):
                    attributes[name] = tuple(swapped) if isinstance(value, tuple) else swapped


def reraise_redacted(exc: BaseException, literals: frozenset[str]) -> None:
    """Mask ``exc`` and raise the masked copy when a group had to be rebuilt."""
    _reraise_forms(exc, _with_escaped(literals) if literals else frozenset())


def _reraise_forms(exc: BaseException, forms: frozenset[str]) -> None:
    """Like reraise_redacted, with the escaped forms already expanded."""
    masked = _redact_forms(exc, forms) if forms else exc
    if masked is exc:
        return
    context = masked.__context__
    try:
        raise masked
    finally:
        # raise in __exit__ chains the handled original, which still holds the raw message.
        object.__setattr__(masked, "__context__", context)


class redact_escaping_exception:  # noqa: N801
    """Mask the current registered literals in any exception leaving the block (a class: see registered_secrets_scope)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        if exc is not None:
            _reraise_forms(exc, current_registered_escaped())


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


_SECRET_KEY = re.compile(r"[\w.-]*" + _SECRET_NAME, re.IGNORECASE)


def _mask_secret_keys(value: Any, ancestors: frozenset[int] = frozenset()) -> Any:
    """Replace the value under every credential-shaped key with ``'***'``, walking mappings, lists and tuples."""
    from mloda.core.abstract_plugins.components.credential import RegisteredCredential  # credential imports us
    from mloda.core.abstract_plugins.components.credential_access import ResolvedCredential

    if isinstance(value, (RegisteredCredential, ResolvedCredential)):
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
