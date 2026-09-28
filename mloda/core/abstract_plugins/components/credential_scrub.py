"""Strip credentials from free text before it reaches core's failure logs or MlodaRunError."""

import re

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
    r"(?P<host>(?:[\w.~-]*|\[[0-9A-Fa-f:.]+\])(?::\d+)?)"
    r"(?P<path>/[\w.~%/:=+-]*)?"
    r"(?:[?#][^\s\"'<>)\]]*)?"
)

_PASSWORD_KEYWORD_PATTERN = re.compile(
    r"(?<![A-Za-z])(?P<keyword>password|passwd|pwd)(?P<sep>\s*=\s*)"
    r"(?:'[^']*'|\"[^\"]*\"|[^;&\s'\"]*)",
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
    """Fallback for unencoded reserved chars in a password: drop everything between `://` and the last `@` before whitespace."""

    def _drop(match: re.Match[str]) -> str:
        scheme, rest = match["scheme"], match["rest"]
        last_at = rest.rfind("@")
        userinfo = rest[:last_at]
        if scheme.lower() in _AZURE_CONTAINER_SCHEMES and _AZURE_CONTAINER_PATTERN.fullmatch(userinfo):
            return match.group(0)
        return f"{scheme}://{rest[last_at + 1 :]}"

    return re.sub(
        r"(?<![A-Za-z0-9+.-])(?P<scheme>(?:jdbc:)?[A-Za-z][A-Za-z0-9+.-]*)://(?P<rest>[^\s]*@[^\s]*)",
        _drop,
        text,
    )


def scrub_credentials(text: str) -> str:
    """Drop URI user info, query and fragment, and mask password= style values. Idempotent."""
    text = _drop_free_text_userinfo(text)
    text = _FREE_TEXT_URI_PATTERN.sub(_scrub_uri_match, text)
    text = _PASSWORD_KEYWORD_PATTERN.sub(r"\g<keyword>\g<sep>***", text)
    return text
