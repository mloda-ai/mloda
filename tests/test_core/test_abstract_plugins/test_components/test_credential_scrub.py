"""Tests for scrub_credentials: strip secrets from free text before it reaches failure logs."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.credential import RegisteredCredential
from mloda.core.abstract_plugins.components.credential_scrub import (
    redact_mapping,
    redact_option_value,
    scrub_credentials,
)
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
    (
        "azure_account_key",
        "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=abc+/hunter2z9==;EndpointSuffix=core.windows.net",
        ["hunter2z9"],
        ["AccountName=acct", "AccountKey=***"],
    ),
    (
        "sas_shared_access_key",
        "Endpoint=sb://ns.servicebus.windows.net/;SharedAccessKeyName=root;SharedAccessKey=hunter2z9",
        ["hunter2z9"],
        ["SharedAccessKeyName=root", "SharedAccessKey=***"],
    ),
    (
        "sas_signature",
        "SharedAccessSignature=sv=2020-08-04&ss=b&sig=hunter2z9",
        ["hunter2z9"],
        ["SharedAccessSignature="],
    ),
    (
        "token_keyword",
        "token=hunter2z9",
        ["hunter2z9"],
        ["token=***"],
    ),
    (
        "client_secret_keyword",
        "client_secret=hunter2z9",
        ["hunter2z9"],
        ["client_secret=***"],
    ),
    (
        "api_key_keyword",
        "api_key=hunter2z9",
        ["hunter2z9"],
        ["api_key=***"],
    ),
    (
        "x_api_key_keyword",
        "X-Api-Key=hunter2z9",
        ["hunter2z9"],
        ["X-Api-Key=***"],
    ),
    (
        "aws_secret_access_key_env_style",
        "AWS_SECRET_ACCESS_KEY=hunter2z9",
        ["hunter2z9"],
        ["AWS_SECRET_ACCESS_KEY=***"],
    ),
    (
        "secret_key_env_style",
        "SECRET_KEY=hunter2z9",
        ["hunter2z9"],
        ["SECRET_KEY=***"],
    ),
    (
        "url_path_token_keyword",
        "https://h/api/token=hunter2z9",
        ["hunter2z9"],
        ["https://h/api"],
    ),
    (
        "nested_quoted_token_dict",
        "{'token': {'access_token': 'hunter2z9'}}",
        ["hunter2z9"],
        ["'token'"],
    ),
    (
        "nested_quoted_token_dict_sibling_key",
        "{'token': {'access_token': 'a', 'other': 'hunter2z9'}}",
        ["hunter2z9"],
        ["'token'"],
    ),
    (
        "nested_quoted_api_key_list",
        "{'api_key': ['k1', 'hunter2z9']}",
        ["hunter2z9"],
        ["'api_key'"],
    ),
    (
        "nested_quoted_password_bytes",
        "{'password': b'a,hunter2z9'}",
        ["hunter2z9"],
        ["'password'"],
    ),
    (
        "quoted_key_after_comma_with_other_key",
        '{"a": 1, "token": "hunter2z9"}',
        ["hunter2z9"],
        ['"a": 1'],
    ),
    (
        "bearer_dotted_token_in_prose",
        "401 with Bearer abc.def.ghi rejected",
        ["abc.def.ghi"],
        ["401 with Bearer *** rejected"],
    ),
    (
        "bearer_lowercase_scheme_digit_token",
        "sent bearer hunter2z9 to host",
        ["hunter2z9"],
        ["bearer ***"],
    ),
    (
        "bearer_uppercase_scheme_jwt_like",
        "BEARER eyJhbGciOi.hunter2z9.sig",
        ["hunter2z9", "eyJhbGciOi"],
        ["BEARER ***"],
    ),
    (
        "bearer_base64_padding",
        "Bearer YWJjZGVm+hunter2z9==",
        ["hunter2z9"],
        ["Bearer ***"],
    ),
    (
        "authorization_basic_header",
        "Authorization: Basic dXNlcjpwYXNz",
        ["dXNlcjpwYXNz"],
        ["Authorization: Basic ***"],
    ),
    (
        "authorization_bearer_header",
        "Authorization: Bearer abc.def.ghi",
        ["abc.def.ghi"],
        ["Authorization: Bearer ***"],
    ),
    (
        "authorization_quoted_dict_bearer_plain_word",
        "{'Authorization': 'Bearer abc'}",
        ["'Bearer abc'"],
        ["{'Authorization': 'Bearer ***"],
    ),
    (
        "authorization_double_quoted_lowercase_key",
        '{"authorization": "Bearer hunter2z9", "host": "h"}',
        ["hunter2z9"],
        ['"authorization": "Bearer ***', '"host": "h"'],
    ),
    (
        "proxy_authorization_basic",
        "Proxy-Authorization: Basic hunter2z9",
        ["hunter2z9"],
        ["Proxy-Authorization: Basic ***"],
    ),
    (
        "authorization_token_scheme_plain_word",
        "Authorization: Token abcdefgh",
        ["abcdefgh"],
        ["Authorization: Token ***"],
    ),
    (
        "authorization_digest_scheme_lowercase",
        "authorization: digest hunterz",
        ["hunterz"],
        ["authorization: digest ***"],
    ),
    (
        "authorization_equals_no_scheme_token_shaped",
        "authorization=eyJhbGciOi.x.y",
        ["eyJhbGciOi"],
        ["authorization="],
    ),
    (
        "access_token_with_bearer_value",
        "access_token=Bearer abc123",
        ["abc123"],
        ["access_token="],
    ),
    (
        "access_token_with_bearer_plain_word",
        "access_token=Bearer hunter",
        ["hunter"],
        ["access_token="],
    ),
    (
        "api_key_with_basic_value",
        "api_key=Basic dXNlcg",
        ["dXNlcg"],
        ["api_key="],
    ),
    (
        "authorization_unknown_scheme_ntlm",
        "Authorization: NTLM TlRMTVNTUAAB",
        ["TlRMTVNTUAAB"],
        ["Authorization: NTLM"],
    ),
    (
        "authorization_unknown_scheme_dpop",
        "Authorization: DPoP eyJhbGciOiJ.x.y",
        ["eyJhbGciOiJ"],
        ["Authorization: DPoP"],
    ),
    (
        "authorization_aws4_signature",
        "Authorization: AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/x, SignedHeaders=host, Signature=abc123def",
        ["AKIDEXAMPLE", "abc123def"],
        ["Authorization: AWS4-HMAC-SHA256"],
    ),
    (
        "authorization_quoted_bearer_value_with_space",
        "{'Authorization': 'Bearer abc def'}",
        ["def"],
        ["{'Authorization': 'Bearer"],
    ),
    (
        "authorization_quoted_digest_params_keep_sibling",
        "{'Authorization': 'Digest username=\"u\", response=\"hunter2z9\"', 'host': 'h'}",
        ["hunter2z9"],
        ["'host': 'h'"],
    ),
    ("over_depth_nesting", "{'password': (a, (b, (c, (d, (e, (f, 'hunter2z9')))))), 'user': 'bob'}", ["hunter2z9"], []),
    ("truncated_text", "{'password': ('a', ('b', 'hunter2z9'", ["hunter2z9"], []),
    ("stray_foreign_closer", "{'password': [a)b, 'hunter2z9']}", ["hunter2z9"], []),
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


DICT_KEY_SCRUB_EXACT_CASES: list[tuple[str, str, str]] = [
    ("dict_repr_password", "{'password': 'hunter2z9'}", "{'password': '***'}"),
    ("dict_repr_numeric_password", "{'db_password': 123456789}", "{'db_password': '***'}"),
    ("dict_repr_password_tuple", "{'password': ('a', 'hunter2z9')}", "{'password': '***'}"),
    ("dict_repr_password_set", "{'password': {'a', 'hunter2z9'}}", "{'password': '***'}"),
    ("dict_repr_token_frozenset", "{'token': frozenset({'a', 'hunter2z9'})}", "{'token': '***'}"),
    (
        "dict_repr_password_tuple_keeps_siblings",
        "{'password': ('a', 'hunter2z9'), 'host': 'h'}",
        "{'password': '***', 'host': 'h'}",
    ),
    ("dict_repr_password_tuple_quoted_paren", "{'password': ('a)b', 'hunter2z9')}", "{'password': '***'}"),
    ("dict_repr_password_list_quoted_bracket", "{'password': ['a]b', 'hunter2z9']}", "{'password': '***'}"),
    ("dict_repr_password_set_quoted_brace", "{'password': {'a}b', 'hunter2z9'}}", "{'password': '***'}"),
    ("dict_repr_token_repr_call_value", "{'token': Secret(value='hunter2z9')}", "{'token': '***'}"),
    ("dict_repr_password_tuple_in_tuple", "{'password': ('a', ('b', 'c'), 'hunter2z9')}", "{'password': '***'}"),
    ("dict_repr_password_list_in_dict", "{'password': {'k': ['a', 'b'], 'z': 'hunter2z9'}}", "{'password': '***'}"),
    ("dict_repr_password_dict_in_list", "{'password': [{'k': 'a'}, 'hunter2z9']}", "{'password': '***'}"),
    (
        "dict_repr_password_nested_tuple_keeps_sibling",
        "{'password': ('a', ('b',), 'x9'), 'user': 'bob'}",
        "{'password': '***', 'user': 'bob'}",
    ),
    (
        "dict_repr_password_nested_quoted_closer",
        "{'password': ('a', (')', 'z'), 's3cr3t')}",
        "{'password': '***'}",
    ),
    (
        "dict_repr_password_three_level_nesting",
        "{'password': ('a', ('b', ['c', 'd']), 'hunter2z9'), 'user': 'bob'}",
        "{'password': '***', 'user': 'bob'}",
    ),
]


def test_scrub_credentials_masks_ordered_dict_value_under_secret_key() -> None:
    assert "hunter2z9" not in scrub_credentials("{'password': OrderedDict([('a', 'hunter2z9')])}")


@pytest.mark.parametrize(
    "case_id,text,expected", DICT_KEY_SCRUB_EXACT_CASES, ids=[case[0] for case in DICT_KEY_SCRUB_EXACT_CASES]
)
def test_scrub_credentials_masks_quoted_dict_keys_exact(case_id: str, text: str, expected: str) -> None:
    assert scrub_credentials(text) == expected


def test_scrub_credentials_masks_json_api_key_keeps_other_keys() -> None:
    text = '{"api_key": "hunter2z9", "host": "h"}'
    result = scrub_credentials(text)
    assert "hunter2z9" not in result
    assert '"host": "h"' in result


SLASH_DSN_CASES: list[tuple[str, str, str, str]] = [
    ("basic_host_port", "scott/hunter2z9@host:1521/service", "hunter2z9", "scott/***@host:1521/service"),
    ("jdbc_thin", "jdbc:oracle:thin:scott/hunter2z9@host:1521:orcl", "hunter2z9", "scott/***@host:1521:orcl"),
    ("ezconnect", "scott/hunter2z9@//host:1521/svc", "hunter2z9", "scott/***@//host:1521/svc"),
    (
        "descriptor",
        "scott/hunter2z9@(DESCRIPTION=(ADDRESS=(HOST=h)))",
        "hunter2z9",
        "scott/***@(DESCRIPTION=(ADDRESS=(HOST=h)))",
    ),
    ("at_in_password", "scott/p@hunter2z9@host:1521/svc", "p@hunter2z9", "scott/***@host:1521/svc"),
    ("ezconnect_no_port", "scott/hunter2z9@host/service", "hunter2z9", "scott/***@host/service"),
]


@pytest.mark.parametrize(
    "case_id,text,secret,expected_substring", SLASH_DSN_CASES, ids=[case[0] for case in SLASH_DSN_CASES]
)
def test_scrub_credentials_masks_slash_dsn_password(
    case_id: str, text: str, secret: str, expected_substring: str
) -> None:
    result = scrub_credentials(text)
    assert secret not in result, f"{case_id}: secret {secret!r} leaked into {result!r}"
    assert expected_substring in result, f"{case_id}: expected {expected_substring!r} in {result!r}"


UNCHANGED_LOOKALIKE_CASES: list[str] = [
    "sort_key=col",
    "primary_key=id",
    "max_tokens=5",
    "tokenizer=bert",
    "key=path/x",
    "SharedAccessKeyName=root",
    "invalid password: too short",
    "path/to/x@y",
    "user@example.com",
    "org/model@main",
    "actions/checkout@v4",
    "library/python@sha256:0123abc",
    "https://s3.amazonaws.com/bucket/report@2024.csv",
    "{'host': 'db1'}",
    "http://localhost:8080/models@v:2",
    "Invalid option 'api_key': expected str, got int",
    "option 'token': must be a str",
    "Missing bearer token",
    "invalid bearer token.",
    "the bearer of bad news",
    "Bearer token required",
    "authorization: denied for role x",
    "authorization failed",
    "Missing Bearer Token",
    "Bearer Token Usage",
    "Authorization: Required",
]


@pytest.mark.parametrize("text", UNCHANGED_LOOKALIKE_CASES)
def test_scrub_credentials_lookalikes_kept_unchanged(text: str) -> None:
    assert scrub_credentials(text) == text


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
        "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=abc+/hunter2z9==;EndpointSuffix=core.windows.net",
        "{'password': 'hunter2z9'}",
        "scott/hunter2z9@host:1521/service",
        "Authorization: Bearer abc.def.ghi",
        "{'Authorization': 'Bearer abc'}",
        "Authorization: Basic dXNlcjpwYXNz",
        "authorization=eyJhbGciOi.x.y",
        "Bearer abc.def.ghi",
        "{'password': ('a', 'hunter2z9')}",
        "{'password': ('a', ('b', 'c'), 'hunter2z9'), 'user': 'bob'}",
        "Authorization: NTLM TlRMTVNTUAAB",
        "Authorization: AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/x, SignedHeaders=host, Signature=abc123def",
        "{'Authorization': 'Digest username=\"u\", response=\"hunter2z9\"', 'host': 'h'}",
        "access_token=Bearer hunter",
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


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        pytest.param("Bearer " + "a" * 50_000, "a" * 100, id="long_run_after_bearer"),
        pytest.param("Authorization: " + "a" * 50_000, None, id="long_run_after_authorization"),
        pytest.param("Bearer " * 20_000, None, id="repeated_bearer_prefix"),
    ],
)
def test_bearer_and_authorization_inputs_scrub_fast(text: str, leaked: str | None) -> None:
    start = time.perf_counter()
    result = scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s"
    if leaked is not None:
        assert leaked not in result


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


def test_dash_run_scrubs_fast() -> None:
    text = "a-" * 25_000
    start = time.perf_counter()
    scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a repeated dash run"


def test_quoted_api_key_repeat_scrubs_fast() -> None:
    text = "'" + "api-key" * 7000
    start = time.perf_counter()
    scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a quoted api-key repeat"


def test_quoted_char_repeat_with_trailing_colon_value_scrubs_fast() -> None:
    text = "'a'" * 20_000 + ": x"
    start = time.perf_counter()
    scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a quoted char repeat with trailing colon value"


def test_slash_run_scrubs_fast() -> None:
    text = "a/" * 25_000
    start = time.perf_counter()
    scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a repeated slash run"


def test_slash_at_repeat_scrubs_fast() -> None:
    text = "a/" + "b@" * 25_000
    start = time.perf_counter()
    scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a slash followed by repeated at run"


def test_slash_password_host_repeat_scrubs_fast() -> None:
    text = "s/p@" + "h" * 50_000
    start = time.perf_counter()
    scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a slash-dsn prefix followed by a long host run"


def test_repeated_unterminated_token_brace_scrubs_fast() -> None:
    text = "token={ " * 6000
    start = time.perf_counter()
    scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on a repeated unterminated token brace"


@pytest.mark.parametrize(
    "text, what",
    [
        ("password={ " * 4400, "a repeated unterminated password brace"),
        ("{'password': ((" + ",'password': ((x" * 5000, "repeated unterminated nested containers"),
        ("{'password': " + "((a)," * 5000 + ")}", "many balanced nested containers"),
    ],
    ids=["unterminated_password_brace", "unterminated_nested_container", "balanced_nested_containers"],
)
def test_repeated_unterminated_password_brace_scrubs_fast(text: str, what: str) -> None:
    start = time.perf_counter()
    scrub_credentials(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scrub_credentials took {elapsed:.3f}s on {what}"


def test_unterminated_password_brace_masks_to_end_of_line() -> None:
    text = "password={a hunter2z9"
    result = scrub_credentials(text)
    assert "hunter2z9" not in result


def test_contained_raise_reason_scrubs_secret_url() -> None:
    exc = OSError("failed for https://u:p@h/k?sig=SECRET")
    result = contained_raise_reason(exc)
    assert result.startswith("raised OSError:")
    assert "SECRET" not in result


def test_redact_mapping_keeps_keys_but_replaces_every_value() -> None:
    _REGISTERED_VALUE = "hunter2plus"  # nosec B105
    result = redact_mapping({"host": "db1", "password": _REGISTERED_VALUE})
    assert result == {"host": "***", "password": "***"}  # nosec B105


def test_redact_mapping_empty_mapping_returns_empty_dict() -> None:
    assert redact_mapping({}) == {}


def test_redact_mapping_non_str_keys_are_preserved() -> None:
    result = redact_mapping({1: "a", ("t",): "b"})
    assert result == {1: "***", ("t",): "***"}


def test_redact_mapping_scrubs_credential_shaped_str_keys() -> None:
    result = redact_mapping({"postgresql://dbuser:hunter2z9@dbhost/db": "x"})
    assert result == {"postgresql://dbhost/db": "***"}


def test_redact_option_value_scrubs_credential_shaped_mapping_key() -> None:
    result = redact_option_value({"postgresql://dbuser:hunter2z9@dbhost/db": "x"})
    assert result == {"postgresql://dbhost/db": "***"}


def test_redact_mapping_str_enum_key_with_nothing_to_scrub_keeps_identity() -> None:
    """A str Enum key with nothing to scrub is kept as the exact same object, not merely equal."""
    result = redact_mapping({_Mode.PLAIN: "value"})
    assert next(iter(result)) is _Mode.PLAIN


class _MarkerReader:
    """Stand-in for a reader class used only as a tuple element in these cases."""


_DSN_URI_MARKER = "postgresql://dbuser:cred_marker_q7@dbhost/db"
_DSN_URI_SCRUBBED = "postgresql://dbhost/db"
_DSN_KEYWORD_MARKER = "host=dbhost password=cred_marker_q7"
_DSN_KEYWORD_SCRUBBED = "host=dbhost password=***"


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"sqlite": "raw_db_path_marker"}, {"sqlite": "***"}),
        (
            RegisteredCredential({"sqlite": "raw_db_path_marker"}),
            {"sqlite": "***"},
        ),
        (
            (_MarkerReader, {"sqlite": "raw_db_path_marker"}),
            (_MarkerReader, {"sqlite": "***"}),
        ),
        ("raw_db_path_marker", "raw_db_path_marker"),
        (["raw_db_path_marker"], ["raw_db_path_marker"]),
        (_DSN_URI_MARKER, _DSN_URI_SCRUBBED),
        (_DSN_KEYWORD_MARKER, _DSN_KEYWORD_SCRUBBED),
        ([{"sqlite": "raw_db_path_marker"}], [{"sqlite": "***"}]),
        (
            (("outer",), {"sqlite": "raw_db_path_marker"}),
            (("outer",), {"sqlite": "***"}),
        ),
        (
            {_DSN_URI_MARKER, "postgresql://dbuser2:cred_marker_q7@dbhost2/db"},
            sorted([_DSN_URI_SCRUBBED, "postgresql://dbhost2/db"], key=repr),
        ),
        (
            frozenset({_DSN_URI_MARKER}),
            [_DSN_URI_SCRUBBED],
        ),
    ],
    ids=[
        "dict",
        "registered_credential",
        "reader_class_tuple",
        "scalar_str",
        "list_unchanged",
        "uri_dsn_str",
        "keyword_dsn_str",
        "list_with_dict",
        "tuple_nested_two_levels_with_dict",
        "set_of_two_uri_dsn_strings",
        "frozenset_single_uri_dsn_string",
    ],
)
def test_redact_option_value(value: Any, expected: Any) -> None:
    assert redact_option_value(value) == expected


def test_redact_option_value_plain_str_returns_same_object() -> None:
    """A non-URI, non-keyword string is returned unchanged by identity."""
    value = "plain configuration text with no credentials"
    assert redact_option_value(value) is value


class _Mode(str, Enum):
    PLAIN = "plain"


def test_redact_option_value_str_enum_member_returns_same_object() -> None:
    """A str Enum member with nothing to scrub is returned as the same object."""
    assert redact_option_value(_Mode.PLAIN) is _Mode.PLAIN


def test_redact_option_value_self_referencing_list_renders_without_recursion_error() -> None:
    """A list that contains itself must not blow the stack; the cycle renders as the literal '<cycle>'."""
    a: list[Any] = [_DSN_URI_MARKER]
    a.append(a)

    result = redact_option_value(a)

    rendered = repr(result)
    assert "cred_marker_q7" not in rendered
    assert "<cycle>" in rendered


def test_redact_option_value_deeply_nested_non_cyclic_list_does_not_raise_recursion_error() -> None:
    """A non-cyclic list nested 5000 levels deep must not blow the stack; past a depth cap it renders '<...>'."""
    value: Any = [_DSN_URI_MARKER]
    for _ in range(5000):
        value = [value]

    result = redact_option_value(value)

    node = result
    while isinstance(node, list):
        node = node[0]
    assert node == "<...>" or "cred_marker_q7" not in str(node)


def test_redact_option_value_shared_inner_dict_without_cycle_renders_both_masked() -> None:
    """The same inner dict referenced twice (no cycle) is masked independently at each position."""
    inner = {"sqlite": "raw_db_path_marker"}
    result = redact_option_value([inner, inner])

    assert result == [{"sqlite": "***"}, {"sqlite": "***"}]
    rendered = repr(result)
    assert "<cycle>" not in rendered
