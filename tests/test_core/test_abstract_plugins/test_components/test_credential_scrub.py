"""Tests for scrub_credentials: strip secrets from free text before it reaches failure logs."""

from __future__ import annotations

import time

import pytest

from mloda.core.abstract_plugins.components.credential_scrub import scrub_credentials
from mloda.core.abstract_plugins.components.utils import contained_raise_reason

# (case id, input text, secret substrings that must be gone, substrings that must survive)
SCRUB_CASES: list[tuple[str, str, list[str], list[str]]] = [
    (
        "presigned_s3_url_in_prose",
        "403 for https://bucket.s3.amazonaws.com/key?X-Amz-Signature=SECRET&X-Amz-Credential=AKIA",
        ["SECRET", "AKIA"],
        ["https://bucket.s3.amazonaws.com/key", "403"],
    ),
    (
        "postgres_userinfo",
        "postgres://user:pw@host:5432/db",
        ["pw", "user:pw"],
        ["postgres://host:5432/db"],
    ),
    (
        "fragment_token",
        "https://h/p#token=SECRET",
        ["SECRET"],
        ["https://h/p"],
    ),
    (
        "unencoded_slash_in_password",
        "postgres://user:p/ss@host/db",
        ["p/ss", "user"],
        ["postgres://host/db"],
    ),
    (
        "unencoded_hash_in_password",
        "postgres://user:p#ss@host/db",
        ["p#ss", "user"],
        ["postgres://host/db"],
    ),
    (
        "unencoded_question_mark_in_password",
        "postgres://user:p?ss@host/db",
        ["p?ss", "user"],
        ["postgres://host/db"],
    ),
    (
        "jdbc_password_param",
        "jdbc:sqlserver://h;user=u;password=SECRET",
        ["SECRET"],
        ["jdbc:sqlserver://h"],
    ),
    (
        "libpq_dsn",
        "host=h password=SECRET dbname=d",
        ["SECRET"],
        ["host=h", "dbname=d", "password="],
    ),
    (
        "libpq_quoted_password",
        "password='a SECRET'",
        ["SECRET"],
        ["password="],
    ),
    (
        "odbc_password_param",
        "Data Source=srv;Password=SECRET;",
        ["SECRET"],
        ["Data Source=srv"],
    ),
    (
        "db_password_env_style",
        "DB_PASSWORD=SECRET",
        ["SECRET"],
        ["DB_PASSWORD="],
    ),
    (
        "pwd_spaced_around_equals",
        "pwd = SECRET",
        ["SECRET"],
        ["pwd"],
    ),
    (
        "key_value_path_segment_cut",
        "https://host/api/id:tok=SECRET/x",
        ["SECRET"],
        ["https://host/api"],
    ),
    (
        "plain_text_without_url",
        "plain error text with no credentials at all",
        [],
        ["plain error text with no credentials at all"],
    ),
    (
        "path_with_parens",
        "https://h/Q1(final).csv?X-Amz-Signature=SECRET",
        ["SECRET"],
        ["https://h/Q1(final).csv"],
    ),
    (
        "path_with_comma",
        "https://h/a,b/c?sig=SECRET",
        ["SECRET"],
        ["https://h/a,b/c"],
    ),
    (
        "path_with_bang",
        "https://h/a!b?sig=SECRET",
        ["SECRET"],
        ["https://h/a!b"],
    ),
    (
        "ipv6_host_query",
        "http://[::1]/a?token=SECRET",
        ["SECRET"],
        ["http://[::1]/a"],
    ),
    (
        "ipv6_userinfo_and_keyword",
        "postgres://u:pw@[::1]:5432/db?sslpassword=SECRET",
        ["pw", "SECRET"],
        ["postgres://[::1]:5432/db"],
    ),
    (
        "sslpassword_keyword",
        "host=h sslpassword=SECRET",
        ["SECRET"],
        ["host=h", "sslpassword="],
    ),
    (
        "pgpassword_env_style",
        "PGPASSWORD=SECRET",
        ["SECRET"],
        ["PGPASSWORD="],
    ),
    (
        "quoted_password_with_escaped_quote",
        "password='a\\' SECRET'",
        ["SECRET"],
        ["password="],
    ),
    (
        "odbc_braced_password",
        "PWD={SECRET;T}",
        ["SECRET"],
        ["PWD="],
    ),
    (
        "unencoded_at_in_password",
        "postgres://user:p@ss@host/db",
        ["p@ss", "user"],
        ["postgres://host/db"],
    ),
]


@pytest.mark.parametrize("case_id,text,secrets,keep", SCRUB_CASES, ids=[case[0] for case in SCRUB_CASES])
def test_scrub_credentials_drops_secrets_and_keeps_context(
    case_id: str, text: str, secrets: list[str], keep: list[str]
) -> None:
    result = scrub_credentials(text)
    for secret in secrets:
        assert secret not in result, f"{case_id}: secret {secret!r} leaked into {result!r}"
    for kept in keep:
        assert kept in result, f"{case_id}: expected {kept!r} kept in {result!r}"


def test_azure_container_uri_kept_unchanged() -> None:
    text = "abfss://container@acct.dfs.core.windows.net/p"
    assert scrub_credentials(text) == text


def test_path_at_symbol_kept_unchanged() -> None:
    text = "https://s3.amazonaws.com/bucket/report@2024.csv"
    assert scrub_credentials(text) == text


def test_prose_at_symbol_kept_unchanged() -> None:
    text = "see https://example.com/a@b for details"
    assert scrub_credentials(text) == text


def test_bare_host_uri_kept_unchanged() -> None:
    text = "postgres://host:5432/db"
    assert scrub_credentials(text) == text


def test_password_keyword_separator_does_not_cross_newline() -> None:
    text = 'password=\n  File "/x/y.py", line 1'
    result = scrub_credentials(text)
    assert 'File "/x/y.py", line 1' in result


def test_multiline_text_lines_after_url_survive_intact() -> None:
    url_line = "connect https://u:pw@host/db?sig=SECRET failed"
    after = "next diagnostic line stays intact"
    text = f"{url_line}\n{after}"
    result = scrub_credentials(text)
    assert "SECRET" not in result
    assert "pw" not in result
    assert after in result


def test_text_without_url_is_unchanged() -> None:
    text = "no secrets here, just a plain failure description"
    assert scrub_credentials(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "postgres://user:pw@host:5432/db",
        "host=h password=SECRET dbname=d",
        "403 for https://bucket.s3.amazonaws.com/key?X-Amz-Signature=SECRET&X-Amz-Credential=AKIA",
    ],
)
def test_scrub_credentials_is_idempotent(text: str) -> None:
    once = scrub_credentials(text)
    twice = scrub_credentials(once)
    assert once == twice


def test_long_alphanumeric_run_scrubs_fast() -> None:
    text = "a" * 50_000
    start = time.perf_counter()
    result = scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a long alphanumeric run"
    assert result == text


def test_long_alphanumeric_run_followed_by_url_scrubs_fast() -> None:
    text = ("a" * 50_000) + " https://u:pw@host/db?sig=SECRET"
    start = time.perf_counter()
    result = scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a long run followed by a url"
    assert "SECRET" not in result
    assert "pw" not in result


def test_repeated_scheme_prefix_scrubs_fast() -> None:
    text = "x://" * 20_000
    start = time.perf_counter()
    scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a repeated scheme prefix"


def test_repeated_jdbc_scheme_prefix_scrubs_fast() -> None:
    text = "jdbc:x://" * 20_000
    start = time.perf_counter()
    scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a repeated jdbc scheme prefix"


def test_repeated_ipv6_bracket_prefix_scrubs_fast() -> None:
    text = "https://[" * 10_000
    start = time.perf_counter()
    scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a repeated ipv6 bracket prefix"


def test_compact_json_url_list_scrubs_fast() -> None:
    urls = ",".join(f'"https://h{i}.example.com/p?sig=SECRET"' for i in range(5000))
    text = f"[{urls}]"
    start = time.perf_counter()
    result = scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a compact json url list"
    assert "SECRET" not in result


def test_long_word_followed_by_password_keyword_scrubs_fast() -> None:
    text = ("a" * 50_000) + " password=SECRET"
    start = time.perf_counter()
    result = scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a long word followed by password="
    assert "SECRET" not in result


def test_contained_raise_reason_scrubs_secret_url() -> None:
    exc = OSError("failed for https://u:p@h/k?sig=SECRET")
    result = contained_raise_reason(exc)
    assert result.startswith("raised OSError:")
    assert "SECRET" not in result
