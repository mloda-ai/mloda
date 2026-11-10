"""URIs with a secret-named parameter region at every chain position and encoding depth."""

import itertools
from collections.abc import Iterator
from urllib.parse import quote, unquote

SECRET = "S3CR3TVAL"  # nosec B105


def encode(sep: str, depth: int) -> str:
    for _ in range(depth):
        sep = quote(sep, safe="")
    return sep


def fixpoint(text: str) -> str:
    for _ in range(20):
        nxt = unquote(text)
        if nxt == text:
            break
        text = nxt
    return text


def region_matrix(suffix: str) -> Iterator[tuple[str, str, str, str]]:
    """Yield (label, uri, expected_identity_source, secret)."""
    plain = {
        "mid": "s3://bucket/year=2024/dirA{r}/my%20dir/data" + suffix,
        "file": "s3://bucket/year=2024/my%20dir/data" + suffix + "{r}",
        "first": "s3://bucket/dirA{r}/year=2024/my%20dir/data" + suffix,
    }
    zipped = {
        "mid": "zip://year=2024/dirA{r}/my%20dir/a.zip::s3://bucket/inner" + suffix,
        "file": "zip://year=2024/my%20dir/a.zip{r}::s3://bucket/inner" + suffix,
        "first": "zip://dirA{r}/year=2024/my%20dir/a.zip::s3://bucket/inner" + suffix,
    }
    chains = {"plain": plain, "zip": zipped}
    for key, sep, assign, depth, position, chain in itertools.product(
        ["token", "auth", "key", "session", "jsessionid", ""],
        [";", "?", "#"],
        ["=", ":", "::", ""],
        range(6),
        ["mid", "file", "first"],
        ["plain", "zip"],
    ):
        template = chains[chain][position]
        uri = template.format(r=encode(sep, depth) + key + assign + SECRET)
        expected = uri[: uri.index(sep)] if depth == 0 and sep in "?#" else template.format(r=";***")
        label = f"key={key!r} sep={sep!r} assign={assign!r} depth={depth} position={position} chain={chain}"
        yield label, uri, expected, SECRET
