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
    strip_uri_userinfo,
)
from mloda.core.abstract_plugins.components.utils import contained_raise_reason, safe_value_text

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
    (
        "x_authorization_basic",
        "X-Authorization: Basic dXNlcjpwYXNz",
        ["dXNlcjpwYXNz"],
        ["X-Authorization: Basic ***"],
    ),
    (
        "x_proxy_authorization_ntlm_equals",
        "X-Proxy-Authorization=NTLM TlRMTVNTUAAB",
        ["TlRMTVNTUAAB"],
        ["X-Proxy-Authorization=NTLM ***"],
    ),
    (
        "http_authorization_basic",
        "HTTP_AUTHORIZATION: Basic dXNlcjpwYXNz",
        ["dXNlcjpwYXNz"],
        ["HTTP_AUTHORIZATION: Basic ***"],
    ),
    (
        "http_authorization_quoted_dict_key",
        "{'HTTP_AUTHORIZATION': 'Basic dXNlcjpwYXNz'}",
        ["dXNlcjpwYXNz"],
        ["{'HTTP_AUTHORIZATION': 'Basic ***"],
    ),
    (
        "http_proxy_authorization_token_shaped",
        "HTTP_PROXY_AUTHORIZATION=eyJhbGciOi.x.y",
        ["eyJhbGciOi"],
        ["HTTP_PROXY_AUTHORIZATION="],
    ),
    (
        "proxy_authorization_underscored_basic",
        "proxy_authorization: Basic dXNlcjpwYXNz",
        ["dXNlcjpwYXNz"],
        ["proxy_authorization: Basic ***"],
    ),
    (
        "authorization_header_underscored_ntlm",
        "authorization_header=NTLM TlRMTVNTUAAB",
        ["TlRMTVNTUAAB"],
        ["authorization_header=NTLM ***"],
    ),
    (
        "x_authorization_header_dashed_basic",
        "X-Authorization-Header: Basic dXNlcjpwYXNz",
        ["dXNlcjpwYXNz"],
        ["X-Authorization-Header: Basic ***"],
    ),
    ("over_depth_nesting", "{'password': (a, (b, (c, (d, (e, (f, 'hunter2z9')))))), 'user': 'bob'}", ["hunter2z9"], []),
    (
        "over_depth_tuple_then_secret",
        "{'password': ('a', ((((('b',),),),),), 'hunter2z9')}",
        ["hunter2z9"],
        [],
    ),
    ("over_depth_list_then_secret", "{'password': ['a', [[[['b']]]], 'hunter2z9']}", ["hunter2z9"], []),
    (
        "over_depth_mixed_tuple_then_secret",
        "{'password': ('a', ('b', ('c', ('d', ('e',), 'x'), 'y'), 'z'), 'hunter2z9')}",
        ["hunter2z9"],
        [],
    ),
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
    "Pre-Authorization: Pending review",
    "payment_authorization: Approved TXN-1234",
    "user_authorization: Denied by policy",
    "card_authorization=ABCDEF123456",
    "the bearer of bad news",
    "Bearer token required",
    "authorization: denied for role x",
    "authorization failed",
    "Missing Bearer Token",
    "Bearer Token Usage",
    "Authorization: Required",
    "author: bob",
    'Authorization-Info: nextnonce="abc123def"',
    "x_authorization: denied for role x",
]


@pytest.mark.parametrize("text", UNCHANGED_LOOKALIKE_CASES)
def test_scrub_credentials_lookalikes_kept_unchanged(text: str) -> None:
    assert scrub_credentials(text) == text


def test_azure_container_uri_kept_unchanged() -> None:
    text = "abfss://container@acct.dfs.core.windows.net/p"
    assert scrub_credentials(text) == text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("s3://alice:hunter2@bucket/my%20file.csv", "s3://bucket/my%20file.csv"),  # nosec B105
        ("s3://AKIA:ab/hunter2@bucket/x.csv", "s3://bucket/x.csv"),  # nosec B105
        ("s3://tok@bucket/x.csv", "s3://bucket/x.csv"),
        ("simplecache::s3://tok@bucket/x.csv", "simplecache::s3://bucket/x.csv"),
        ("jdbc:hive2://tok@host/x.csv", "jdbc:hive2://host/x.csv"),
        ("s3://bucket/a@b.csv", "s3://bucket/a@b.csv"),
        ("s3://bucket:9000/x.csv", "s3://bucket:9000/x.csv"),
        ("abfss://cont@acct.dfs.core.windows.net/a%20b.csv", "abfss://cont@acct.dfs.core.windows.net/a%20b.csv"),
        ("s3://bucket]/x.csv", "s3://bucket]/x.csv"),
    ],
)
def test_strip_uri_userinfo(text: str, expected: str) -> None:
    assert strip_uri_userinfo(text) == expected


def test_strip_uri_userinfo_never_raises_on_a_bad_bracket_host() -> None:
    assert isinstance(strip_uri_userinfo("s3://[bad/x.csv"), str)


def test_strip_uri_userinfo_strips_every_uri_in_the_text() -> None:
    assert strip_uri_userinfo("s3://a:p1@b/x.csv::gs://c:p2@d/y.csv") == "s3://b/x.csv::gs://d/y.csv"  # nosec B105


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
        "HTTP_AUTHORIZATION: Basic dXNlcjpwYXNz",
        "X-Proxy-Authorization=NTLM TlRMTVNTUAAB",
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
        pytest.param("x_" * 50_000 + "authorization", None, id="long_prefix_run_before_authorization"),
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
        ("{'password': " + "((((((a," * 5000, "repeated over-depth unterminated containers"),
    ],
    ids=[
        "unterminated_password_brace",
        "unterminated_nested_container",
        "balanced_nested_containers",
        "over_depth_unterminated_containers",
    ],
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


class TestScrubUriProjection:
    """A fullmatching scheme://... drops user info, query and fragment. A jdbc: scheme drops the path entirely;
    any other scheme cuts its path back to the last / before the first percent-escape, and fails closed when a
    path segment has a : followed by =. An azure container userinfo is kept only when it fullmatches an azure
    container name or a well-known $root/$web/$logs alias."""

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            pytest.param(
                "postgresql://admin:s3cr3t@host:5432/db",
                "postgresql://host:5432/db",
                id="userinfo-stripped",
            ),
            pytest.param(
                "abfss://key:s3cr3t@account.dfs.core.windows.net/p",
                "abfss://account.dfs.core.windows.net/p",
                id="abfss-userinfo-with-colon-is-stripped",
            ),
            pytest.param("postgresql://u:p@ss@host/db", "postgresql://host/db", id="at-sign-in-userinfo"),
            pytest.param("postgresql://u:p@host", "postgresql://host", id="no-path"),
            pytest.param(
                "abfss://container@account.dfs.core.windows.net/p?sig=S",
                "abfss://container@account.dfs.core.windows.net/p",
                id="abfss-keeps-container-drops-query",
            ),
            pytest.param(
                "abfs://container@account.dfs.core.windows.net/p",
                "abfs://container@account.dfs.core.windows.net/p",
                id="abfs-keeps-container",
            ),
            pytest.param(
                "wasb://container@account.blob.core.windows.net/p",
                "wasb://container@account.blob.core.windows.net/p",
                id="wasb-keeps-container",
            ),
            pytest.param(
                "wasbs://container@account.blob.core.windows.net/p",
                "wasbs://container@account.blob.core.windows.net/p",
                id="wasbs-keeps-container",
            ),
            pytest.param(
                "ABFSS://container@account.dfs.core.windows.net/p",
                "ABFSS://container@account.dfs.core.windows.net/p",
                id="upper-case-abfss-keeps-container",
            ),
            pytest.param(
                "WASB://container@account.blob.core.windows.net/p",
                "WASB://container@account.blob.core.windows.net/p",
                id="upper-case-wasb-keeps-container",
            ),
            pytest.param(
                "wasbs://key:secret@account.blob.core.windows.net/p",
                "wasbs://account.blob.core.windows.net/p",
                id="wasbs-userinfo-with-colon-is-stripped",
            ),
            pytest.param(
                "wasbs://container@account.blob.core.windows.net/p?sig=S",
                "wasbs://container@account.blob.core.windows.net/p",
                id="wasbs-keeps-container-drops-query",
            ),
            pytest.param("postgresql://token@host/db", "postgresql://host/db", id="non-azure-colon-free-userinfo"),
            pytest.param(
                "postgresql://host/db%3Fpassword=hunter2",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-question-mark-secret",
            ),
            pytest.param(
                "postgresql://host/db%3fpassword=hunter2",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-lower-case-question-mark-secret",
            ),
            pytest.param(
                "postgresql://host/db%23password=hunter2",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-hash-secret",
            ),
            pytest.param(
                "postgresql+psycopg2://u:p@host/db",
                "postgresql+psycopg2://host/db",
                id="scheme-with-plus-suffix",
            ),
            pytest.param(
                "https://host/a%3Fb.csv",
                "https://host/",
                id="path-cut-back-to-segment-before-percent-encoded-question-mark",
            ),
            pytest.param(
                "postgresql://host/db%3Flimit=10",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-question-mark-with-non-secret-key",
            ),
            pytest.param("file:///tmp/a.csv", "file:///tmp/a.csv", id="file-uri-with-empty-host"),
            pytest.param("sqlite:///x.db", "sqlite:///x.db", id="sqlite-uri-with-empty-host"),
            pytest.param(
                "s3://bucket/year=2024/part.parquet",
                "s3://bucket/year=2024/part.parquet",
                id="s3-uri-with-equals-in-path",
            ),
            pytest.param("http://[::1]/x", "http://[::1]/x", id="ipv6-host-without-at-sign"),
            pytest.param(
                "jdbc:postgresql://u:p@host:5432/db?ssl=true",
                "jdbc:postgresql://host:5432",
                id="jdbc-scheme-drops-path-entirely",
            ),
            pytest.param(
                "jdbc:db2://host:50000/DB:password=hunter2",
                "jdbc:db2://host:50000",
                id="jdbc-db2-drops-path-with-secret",
            ),
            pytest.param(
                "jdbc:teradata://host/PASSWORD=hunter2",
                "jdbc:teradata://host",
                id="jdbc-teradata-drops-path-with-secret",
            ),
            pytest.param(
                "jdbc:informix-sqli://h:1533/db:password=hunter2",
                "jdbc:informix-sqli://h:1533",
                id="jdbc-informix-drops-path-with-secret",
            ),
            pytest.param(
                "postgresql://host/db%3Bpassword=hunter2",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-semicolon",
            ),
            pytest.param(
                "postgresql://host/db%26password=hunter2",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-ampersand",
            ),
            pytest.param(
                "https://host/data%253Fpassword%253Dhunter2",
                "https://host/",
                id="path-cut-back-to-segment-before-double-percent-encoded-escape",
            ),
            pytest.param(
                "s3://bucket/my%20file.csv",
                "s3://bucket/",
                id="path-cut-back-to-segment-before-percent-encoded-space",
            ),
            pytest.param(
                "s3://bucket/2024%2F01/a.parquet",
                "s3://bucket/",
                id="path-cut-back-to-segment-before-percent-encoded-slash",
            ),
            pytest.param(
                "s3://bucket/dir/my%20file.csv",
                "s3://bucket/dir/",
                id="path-cut-back-to-parent-dir-before-percent-escape",
            ),
            pytest.param(
                "https://host/a/b%20c/d.csv",
                "https://host/a/",
                id="path-cut-back-to-segment-before-mid-path-percent-escape",
            ),
            pytest.param("file:///C:/data/x.csv", "file:///C:/data/x.csv", id="windows-drive-colon-in-file-uri-kept"),
            pytest.param(
                "s3://bucket/2024-01-01T00:00:00/part.parquet",
                "s3://bucket/2024-01-01T00:00:00/part.parquet",
                id="timestamp-colons-in-path-segment-kept",
            ),
            pytest.param(
                "s3://bucket/ts=2024-01-01T00:00:00/part.parquet",
                "s3://bucket/ts=2024-01-01T00:00:00/part.parquet",
                id="equals-before-colon-hive-partition-kept",
            ),
            pytest.param(
                "abfss://$web@account.dfs.core.windows.net/p",
                "abfss://$web@account.dfs.core.windows.net/p",
                id="well-known-dollar-container-alias-is-kept",
            ),
            pytest.param(
                "abfss://Container_X@account.dfs.core.windows.net/p",
                "abfss://account.dfs.core.windows.net/p",
                id="upper-case-and-underscore-container-is-dropped",
            ),
            pytest.param(
                "abfss://ab@account.dfs.core.windows.net/p",
                "abfss://account.dfs.core.windows.net/p",
                id="too-short-container-is-dropped",
            ),
        ],
    )
    def test_scrub_keeps_host_and_path_only(self, uri: str, expected: str) -> None:
        assert scrub_credentials(uri) == expected

    def test_long_uri_without_secrets_returns_quickly(self) -> None:
        value = "s3://b/" + "a" * 40000
        assert scrub_credentials(value) == value

    def test_long_uri_with_a_trailing_invalid_character_returns_quickly(self) -> None:
        value = "a://" + "a@" * 20000 + "/" + "a" * 20000 + "!"
        started = time.monotonic()
        scrub_credentials(value)
        assert time.monotonic() - started < 5


class TestScrubUriFailClosed:
    """A URI whose path or userinfo does not project cleanly never keeps its secret."""

    @pytest.mark.parametrize(
        ("value", "secret"),
        [
            pytest.param("s3://bucket/key with space password=hunter2", "hunter2", id="space-in-uri-path"),
            pytest.param("postgresql://host/db\npassword=hunter2", "hunter2", id="newline-in-uri-path"),
            pytest.param("https://host/db&password=hunter2", "hunter2", id="ampersand-in-uri-path"),
            pytest.param("s3://bucket/path;password=hunter2", "hunter2", id="semicolon-in-uri-path"),
            pytest.param(
                "jdbc:sqlserver://host:1433;databaseName=db;user=a;password=hunter2",
                "hunter2",
                id="jdbc-sqlserver-semicolons",
            ),
            pytest.param("jdbc:db2://host:50000/DB:user=u;password=hunter2;", "hunter2", id="jdbc-db2-semicolon"),
            pytest.param("mongodb://host/db:password=hunter2", "hunter2", id="colon-then-equals-in-path-segment"),
            pytest.param(
                "https://host/a/db:password=hunter2/x.csv", "hunter2", id="colon-then-equals-in-mid-path-segment"
            ),
            pytest.param("postgresql://u:12?ss@host/db", "ss", id="colon-digit-question-mark-in-userinfo"),
            pytest.param("DRIVER={x};Server=https://host;PWD=secret", "secret", id="keyword-string-with-scheme"),
        ],
    )
    def test_secret_is_not_kept(self, value: str, secret: str) -> None:
        assert secret not in scrub_credentials(value)


def test_safe_value_text_scrubs_a_credential_str_and_keeps_the_type() -> None:
    text = safe_value_text("postgres://u:hunter2@h/db")  # nosec B105
    assert "hunter2" not in text
    assert text.startswith("str ")
    assert safe_value_text("plain") == "str 'plain'"


_LITERAL = "Zq81Lm5Vx0Rb7Ye3Hw29"  # nosec B105
_OTHER_LITERAL = "Ab12Cd34Ef56Gh78Ij90"  # nosec B105


class TestRegisteredSecretsScope:
    """registered_secrets_scope adds exact literals to scrub_credentials for the current context only."""

    def test_outside_any_scope_output_is_unchanged(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import current_registered_secrets

        text = f"username={_LITERAL}"
        assert current_registered_secrets() == frozenset()
        assert scrub_credentials(text) == text

    def test_inside_a_scope_the_literal_is_masked(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

        with registered_secrets_scope(frozenset({_LITERAL})):
            masked = scrub_credentials(f"username={_LITERAL} done")

        assert _LITERAL not in masked
        assert "***" in masked

    def test_nested_scopes_union_and_reset_in_order(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import (
            current_registered_secrets,
            registered_secrets_scope,
        )

        with registered_secrets_scope(frozenset({_LITERAL})):
            with registered_secrets_scope(frozenset({_OTHER_LITERAL})):
                assert current_registered_secrets() == frozenset({_LITERAL, _OTHER_LITERAL})
                masked = scrub_credentials(f"{_LITERAL} {_OTHER_LITERAL}")
                assert _LITERAL not in masked
                assert _OTHER_LITERAL not in masked
            assert current_registered_secrets() == frozenset({_LITERAL})

        assert current_registered_secrets() == frozenset()

    def test_the_set_resets_after_an_exception(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import (
            current_registered_secrets,
            registered_secrets_scope,
        )

        with pytest.raises(RuntimeError):
            with registered_secrets_scope(frozenset({_LITERAL})):
                raise RuntimeError("inside")

        assert current_registered_secrets() == frozenset()

    def test_text_already_containing_the_old_placeholder_word_survives(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

        with registered_secrets_scope(frozenset({_LITERAL})):
            masked = scrub_credentials(f"MLODAREDACTED then {_LITERAL}")

        assert masked == "MLODAREDACTED then ***"

    def test_data_access_collection_redact_keeps_the_old_placeholder_word(self) -> None:
        from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection

        dac = DataAccessCollection(credentials={"h1": {"token": _LITERAL}})

        assert dac.redact(f"MLODAREDACTED then {_LITERAL}") == "MLODAREDACTED then ***"


def test_mask_literals_masks_every_occurrence_and_merges_overlaps() -> None:
    from mloda.core.abstract_plugins.components.credential_scrub import mask_literals

    masked = mask_literals(f"a {_LITERAL} b {_LITERAL}", {_LITERAL})
    assert _LITERAL not in masked
    assert masked.startswith("a ")
    assert mask_literals("nothing here", {_LITERAL}) == "nothing here"


class TestRedactException:
    """redact_exception rewrites registered literals in place, across the chain, keeping identity."""

    def test_args_cause_and_context_are_rewritten_and_identity_kept(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        cause = KeyError(f"k {_LITERAL}")
        context = LookupError(f"c {_LITERAL}")
        exc = ValueError(f"u={_LITERAL}", 7)
        exc.__cause__ = cause
        exc.__context__ = context

        result = redact_exception(exc, frozenset({_LITERAL}))

        assert result is exc
        assert exc.__cause__ is cause
        assert exc.__context__ is context
        assert exc.args[1] == 7
        assert _LITERAL not in str(exc)
        assert _LITERAL not in str(cause)
        assert _LITERAL not in str(context)

    def test_a_cycle_terminates(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        first = ValueError(_LITERAL)
        second = ValueError(_LITERAL)
        first.__cause__ = second
        second.__context__ = first

        redact_exception(first, frozenset({_LITERAL}))

        assert _LITERAL not in str(first)
        assert _LITERAL not in str(second)

    def test_oserror_fields_and_notes_are_rewritten(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        exc = OSError(2, f"bad {_LITERAL}", f"/x/{_LITERAL}")
        object.__setattr__(exc, "__notes__", [f"note {_LITERAL}"])

        redact_exception(exc, frozenset({_LITERAL}))

        assert exc.errno == 2
        assert _LITERAL not in str(exc)
        assert _LITERAL not in str(exc.strerror)
        assert _LITERAL not in str(exc.filename)
        assert all(_LITERAL not in note for note in getattr(exc, "__notes__"))

    def test_a_frozen_dataclass_exception_is_rewritten(self) -> None:
        from dataclasses import dataclass

        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        @dataclass(frozen=True)
        class FrozenError(Exception):
            detail: str

        exc = FrozenError(f"d {_LITERAL}")

        redact_exception(exc, frozenset({_LITERAL}))

        assert type(exc) is FrozenError
        assert _LITERAL not in str(exc)

    def test_resolution_error_refusals_are_rewritten(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception
        from mloda.core.prepare.identify_feature_group import FeatureResolutionError

        refusal = PermissionError(f"denied {_LITERAL}")
        error = FeatureResolutionError(f"msg {_LITERAL}", "f", None, refusals=[refusal])  # type: ignore[arg-type]

        redact_exception(error, frozenset({_LITERAL}))

        assert _LITERAL not in str(error)
        assert all(_LITERAL not in str(item) for item in error.refusals)
        assert error.refusals[0] is refusal

    def test_a_frozen_dataclass_exception_with_notes_is_rewritten(self) -> None:
        from dataclasses import dataclass

        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        @dataclass(frozen=True)
        class FrozenNotedError(Exception):
            detail: str

        exc = FrozenNotedError(f"d {_LITERAL}")
        object.__setattr__(exc, "__notes__", [f"note {_LITERAL}"])

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in str(exc)
        assert all(_LITERAL not in note for note in getattr(exc, "__notes__"))

    @pytest.mark.parametrize("attributes", [{"exceptions": 3}, {"exceptions": ["a", "b"]}, {"refusals": None}])
    def test_unrelated_exceptions_or_refusals_attributes_are_ignored(self, attributes: dict[str, Any]) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        class OddError(Exception):
            pass

        exc = OddError(f"u={_LITERAL}")
        for name, value in attributes.items():
            setattr(exc, name, value)

        result = redact_exception(exc, frozenset({_LITERAL}))

        assert result is exc
        assert _LITERAL not in str(exc)

    def test_no_literals_leaves_the_exception_untouched(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        exc = ValueError(f"keep {_LITERAL}")

        redact_exception(exc, frozenset())

        assert _LITERAL in str(exc)

    def test_a_str_nested_in_a_tuple_arg_is_masked_keeping_the_container_type(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        exc = KeyError(("h1", _LITERAL))

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in repr(exc.args)
        assert _LITERAL not in str(exc)
        assert exc.args == (("h1", "***"),)

    def test_a_str_nested_in_a_dict_arg_is_masked(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        exc = ValueError("bad", {"cfg": [_LITERAL]})

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in repr(exc.args)
        assert exc.args == ("bad", {"cfg": ["***"]})

    def test_a_self_referencing_arg_terminates(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        loop: list[Any] = [_LITERAL]
        loop.append(loop)
        exc = ValueError(loop)

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in repr(exc.args)

    def test_a_str_instance_attribute_is_masked(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        class TokenHolderError(Exception):
            def __init__(self, token: str) -> None:
                super().__init__("failed")
                self.token = token

            def __str__(self) -> str:
                return self.token

        exc = TokenHolderError(_LITERAL)

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in str(exc)
        assert exc.token == "***"  # nosec B105

    def test_a_frozen_dataclass_exception_repr_is_masked(self) -> None:
        from dataclasses import dataclass

        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        @dataclass(frozen=True)
        class FrozenDetailError(Exception):
            detail: str

        exc = FrozenDetailError(f"d {_LITERAL}")

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in repr(exc)
        assert _LITERAL not in str(exc)
