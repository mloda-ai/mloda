"""Tests for scrub_credentials: strip secrets from free text before it reaches failure logs."""

from __future__ import annotations

import json
import os
import pickle  # nosec B403
import time
import traceback
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple
from urllib.parse import quote

import pytest

from mloda.core.abstract_plugins.components.credential import RegisteredCredential
from mloda.core.abstract_plugins.components.credential_scrub import (
    _SECRET_KEY,
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
    ("jsessionid_path_param", "https://host/x;jsessionid=SECRET", ["SECRET"], ["https://host/x;jsessionid=***"]),
    ("phpsessid_keyword", "PHPSESSID=SECRET", ["SECRET"], ["PHPSESSID=***"]),
    ("sessionid_quoted_dict", "{'sessionid': 'SECRET'}", ["SECRET"], ["'sessionid'"]),
    ("aspnet_sessionid_keyword", "ASP.NET_SessionId=SECRET", ["SECRET"], ["ASP.NET_SessionId="]),
    (
        "percent_encoded_bearer_token",
        "https://host/api/Bearer%20eyJhbGciOiJIUzI1NiJ9.abc",
        ["eyJhbGciOiJIUzI1NiJ9"],
        ["https://host/api/"],
    ),
    (
        "password_in_scheme_position",
        "alice:hunter2://host/db",  # nosec B105
        ["hunter2"],  # nosec B105
        ["alice:***://host/db"],
    ),
    (
        "slash_in_userinfo",
        "https://tok/en@host/x",
        ["tok/en@"],
        ["https://tok/***@host/x"],
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
    ("private_use_char_in_input_is_replaced", "a\ue001bcd rest", "a\ufffdbcd rest"),
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
    "session_id=abc",
    "session-id=abc",
    "Source:s3://bucket/x.csv",
    "see:https://example.com/a",
    "https://unpkg.com/react@18/umd/x.js",
    "https://github.com/user@host/x",
    "jdbc:h2:tcp://host/db",
    "filecache::simplecache::s3://bucket/x.csv",
    "INFO:mloda:https://example.com/a",
    "INFO:root:s3://bucket/x.csv",
    "WARNING:urllib3:http://h",
    "Source:jdbc:postgresql://host/db",
    "url:snowflake://acct/db",
    "connect:mssql+pyodbc://host/db",
    "uri:mongodb+srv://cluster/db",
    "backend:sqlite:///tmp/x.db",
    "Location:wss://x/y",
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
        ("jdbc:h2:tcp://u:pw@host/db", "jdbc:h2:tcp://host/db"),  # nosec B105
        ("s3://bucket/a@b.csv", "s3://bucket/a@b.csv"),
        ("s3://bucket:9000/x.csv", "s3://bucket:9000/x.csv"),
        ("abfss://cont@acct.dfs.core.windows.net/a%20b.csv", "abfss://cont@acct.dfs.core.windows.net/a%20b.csv"),
        ("s3://bucket]/x.csv", "s3://bucket]/x.csv"),
        ("https://user:123/PW@host/x", "https://user:123/PW@host/x"),  # nosec B105
        ("https://user:123/PW@host:8080/x", "https://user:123/PW@host:8080/x"),  # nosec B105
        ("https://evil.example:8443/x@trusted.corp/data.csv", "https://evil.example:8443/x@trusted.corp/data.csv"),
        ("https://host:8443/a@b.csv", "https://host:8443/a@b.csv"),
    ],
)
def test_strip_uri_userinfo(text: str, expected: str) -> None:
    assert strip_uri_userinfo(text) == expected


@pytest.mark.parametrize(
    ("key", "masked"),
    [
        ("sessionid", True),
        ("sessionId", True),
        ("PHPSESSID", True),
        ("session_id", False),
        ("session-id", False),
    ],
)
def test_secret_key_session_id_names(key: str, masked: bool) -> None:
    assert (_SECRET_KEY.fullmatch(key) is not None) is masked


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
        "alice:hunter2://host/db",
        "https://tok/en@host/x",
    ],
)
def test_scrub_credentials_is_idempotent(text: str) -> None:
    once = scrub_credentials(text)
    twice = scrub_credentials(once)
    assert once == twice


def _assert_scrubs_linearly(build: Callable[[int], str], n: int, what: str) -> str:
    """Fail if min CPU time at n reaches 8x that at n // 4 (small floored at 2 ms) or 2 s; linear is ~4x, quadratic ~16x."""
    small, large = build(n // 4), build(n)
    min_small = min_large = float("inf")
    result = ""
    ratio = float("inf")
    for _ in range(3):
        start = time.process_time()
        scrub_credentials(small)
        min_small = min(min_small, time.process_time() - start)
        start = time.process_time()
        result = scrub_credentials(large)
        min_large = min(min_large, time.process_time() - start)
        ratio = min_large / max(min_small, 2e-3)
        if ratio < 8.0:
            break
    assert ratio < 8.0, (
        f"scrub_credentials scales super-linearly on {what}: "
        f"min small {min_small:.4f}s, min large {min_large:.4f}s, ratio {ratio:.1f}"
    )
    assert min_large < 2.0, f"scrub_credentials is too slow on {what}: min large {min_large:.4f}s"
    return result


def test_long_alphanumeric_run_scrubs_fast() -> None:
    result = _assert_scrubs_linearly(lambda k: "a" * k, 50_000, "a long alphanumeric run")
    assert result == "a" * 50_000


@pytest.mark.parametrize(
    ("build", "n", "leaked"),
    [
        pytest.param(lambda k: "Bearer " + "a" * k, 50_000, "a" * 100, id="long_run_after_bearer"),
        pytest.param(lambda k: "Authorization: " + "a" * k, 50_000, None, id="long_run_after_authorization"),
        pytest.param(lambda k: "Bearer " * k, 20_000, None, id="repeated_bearer_prefix"),
        pytest.param(lambda k: "x_" * k + "authorization", 50_000, None, id="long_prefix_run_before_authorization"),
    ],
)
def test_bearer_and_authorization_inputs_scrub_fast(build: Callable[[int], str], n: int, leaked: str | None) -> None:
    result = _assert_scrubs_linearly(build, n, "a bearer or authorization input")
    if leaked is not None:
        assert leaked not in result


def test_long_alphanumeric_run_followed_by_url_scrubs_fast() -> None:
    result = _assert_scrubs_linearly(
        lambda k: ("a" * k) + " https://u:pw@host/db?sig=SECRET", 50_000, "a long run followed by a url"
    )
    assert "SECRET" not in result
    assert "pw" not in result


def test_repeated_scheme_prefix_scrubs_fast() -> None:
    _assert_scrubs_linearly(lambda k: "x://" * k, 20_000, "a repeated scheme prefix")


@pytest.mark.parametrize(
    ("build", "n"),
    [
        pytest.param(lambda k: "jdbc:x://" * k, 20_000, id="jdbc_x_slashes"),
        pytest.param(lambda k: "jdbc:x:" * k, 20_000, id="jdbc_x_colon"),
        pytest.param(lambda k: "jdbc:" * k, 20_000, id="jdbc_only"),
    ],
)
def test_repeated_jdbc_scheme_prefix_scrubs_fast(build: Callable[[int], str], n: int) -> None:
    _assert_scrubs_linearly(build, n, "a repeated jdbc scheme prefix")


@pytest.mark.parametrize(
    ("build", "n"),
    [
        pytest.param(lambda k: "a~" * k, 20_000, id="tilde"),
        pytest.param(lambda k: "a:" * k + "://h", 20_000, id="colon_then_scheme"),
        pytest.param(lambda k: "https://" + "a/b@" * k, 20_000, id="slash_at"),
        pytest.param(lambda k: "(a:" * k, 20_000, id="paren"),
        pytest.param(lambda k: "[x" * k, 20_000, id="bracket"),
        pytest.param(lambda k: "https://h/x?a=1)" * k, 20_000, id="url_query_close_paren_no_space"),
        pytest.param(lambda k: "a:b:c://" * k, 20_000, id="two_colon_scheme_repeat"),
        pytest.param(lambda k: "postgresql://" + "u:12/" * k, 20_000, id="db_digit_slash_without_at"),
        pytest.param(lambda k: "postgresql://" + "u:12/" * k + "@", 20_000, id="db_digit_slash_with_at"),
        pytest.param(lambda k: "zip://d;%74oken=x'" * k, 5000, id="encoded_host_param_quote_repeat"),
        pytest.param(lambda k: "zip://d%74oken=x " * k, 5000, id="encoded_host_key_repeat"),
        pytest.param(lambda k: "postgresql://u:5432/" * k, 20_000, id="db_port_slash_repeat"),
        pytest.param(lambda k: "k:v," * k, 20_000, id="kv_comma"),
        pytest.param(lambda k: "a:b," * k, 20_000, id="ab_comma"),
        pytest.param(lambda k: "k=a:b;" * k, 20_000, id="kv_semicolon"),
        pytest.param(lambda k: "tags=" + ";".join(f"env{i}:prod" for i in range(k)), 6000, id="tags"),
    ],
)
def test_repeated_userinfo_shaped_prefix_scrubs_fast(build: Callable[[int], str], n: int) -> None:
    _assert_scrubs_linearly(build, n, "a repeated userinfo-shaped prefix")


# A literal whose raw, repr, ASCII JSON and UTF-8 JSON forms all differ (quote, backslash, non-ASCII).
_FORM_LITERAL = 'sec"re\\t-ä-1234'  # nosec B105
_FORM_BUILDERS: dict[str, Callable[[str], str]] = {
    "raw": lambda literal: literal,
    "repr": lambda literal: repr(literal)[1:-1],
    "json_ascii": lambda literal: json.dumps(literal)[1:-1],
    "json_utf8": lambda literal: json.dumps(literal, ensure_ascii=False)[1:-1],
}


def test_the_form_literal_has_four_distinct_forms() -> None:
    assert len({build(_FORM_LITERAL) for build in _FORM_BUILDERS.values()}) == 4


@pytest.mark.parametrize("form", list(_FORM_BUILDERS))
def test_scrub_credentials_masks_every_form_of_a_registered_literal(form: str) -> None:
    from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

    echoed = _FORM_BUILDERS[form](_FORM_LITERAL)
    with registered_secrets_scope(frozenset({_FORM_LITERAL})):
        result = scrub_credentials(f"login {echoed} ok")

    assert result == "login *** ok"


@pytest.mark.parametrize("form", list(_FORM_BUILDERS))
def test_scrub_credentials_cuts_a_percent_encoded_path_hiding_any_form(form: str) -> None:
    from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

    literal = "ab\\cd-ä-12345"  # nosec B105
    encoded = quote(_FORM_BUILDERS[form](literal), safe="")
    with registered_secrets_scope(frozenset({literal})):
        result = scrub_credentials(f"https://host/dir/{encoded}")

    assert encoded not in result
    assert result.startswith("https://host/dir/")


def test_repeated_escaped_form_with_a_registered_literal_scrubs_fast() -> None:
    from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

    escaped = _FORM_BUILDERS["json_ascii"](_FORM_LITERAL)
    with registered_secrets_scope(frozenset({_FORM_LITERAL})):
        result = _assert_scrubs_linearly(lambda k: f"v {escaped} " * k, 8000, "a repeated escaped secret form")

    assert escaped not in result


def test_repeated_password_keyword_with_registered_literals_scrubs_fast() -> None:
    from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

    with registered_secrets_scope(frozenset({"Zq81Lm5Vx0Rb7Ye3Hw29", "password"})):
        _assert_scrubs_linearly(lambda k: "password=x " * k, 20_000, "a repeated password keyword with literals")


def test_repeated_ipv6_bracket_prefix_scrubs_fast() -> None:
    _assert_scrubs_linearly(lambda k: "https://[" * k, 10_000, "a repeated ipv6 bracket prefix")


def test_compact_json_url_list_scrubs_fast() -> None:
    def build(k: int) -> str:
        urls = ",".join(f'"https://h{i}.example.com/p?sig=SECRET"' for i in range(k))
        return f"[{urls}]"

    result = _assert_scrubs_linearly(build, 5000, "a compact json url list")
    assert "SECRET" not in result


def test_long_word_followed_by_password_keyword_scrubs_fast() -> None:
    result = _assert_scrubs_linearly(
        lambda k: ("a" * k) + " password=SECRET", 50_000, "a long word followed by password="
    )
    assert "SECRET" not in result


def test_dash_run_scrubs_fast() -> None:
    _assert_scrubs_linearly(lambda k: "a-" * k, 25_000, "a repeated dash run")


def test_quoted_api_key_repeat_scrubs_fast() -> None:
    _assert_scrubs_linearly(lambda k: "'" + "api-key" * k, 7000, "a quoted api-key repeat")


def test_quoted_char_repeat_with_trailing_colon_value_scrubs_fast() -> None:
    _assert_scrubs_linearly(lambda k: "'a'" * k + ": x", 20_000, "a quoted char repeat with trailing colon value")


def test_slash_run_scrubs_fast() -> None:
    _assert_scrubs_linearly(lambda k: "a/" * k, 25_000, "a repeated slash run")


def test_slash_at_repeat_scrubs_fast() -> None:
    _assert_scrubs_linearly(lambda k: "a/" + "b@" * k, 25_000, "a slash followed by repeated at run")


def test_slash_password_host_repeat_scrubs_fast() -> None:
    _assert_scrubs_linearly(lambda k: "s/p@" + "h" * k, 50_000, "a slash-dsn prefix followed by a long host run")


def test_repeated_unterminated_token_brace_scrubs_fast() -> None:
    _assert_scrubs_linearly(lambda k: "token={ " * k, 6000, "a repeated unterminated token brace")


@pytest.mark.parametrize(
    "build, n, what",
    [
        (lambda k: "password={ " * k, 4400, "a repeated unterminated password brace"),
        (lambda k: "{'password': ((" + ",'password': ((x" * k, 5000, "repeated unterminated nested containers"),
        (lambda k: "{'password': " + "((a)," * k + ")}", 5000, "many balanced nested containers"),
        (lambda k: "{'password': " + "((((((a," * k, 5000, "repeated over-depth unterminated containers"),
    ],
    ids=[
        "unterminated_password_brace",
        "unterminated_nested_container",
        "balanced_nested_containers",
        "over_depth_unterminated_containers",
    ],
)
def test_repeated_unterminated_password_brace_scrubs_fast(build: Callable[[int], str], n: int, what: str) -> None:
    _assert_scrubs_linearly(build, n, what)


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
    """A fullmatching scheme://... drops user info, query and fragment. A jdbc: path is dropped when it has =, ; or @,
    else handled like any other scheme; any other scheme keeps a percent-escaped path unless its decoded form is
    malformed or still scrubs (then it cuts back to the last / before the first %), and fails closed when a path
    segment has a : followed by =. An azure container userinfo is kept only when it fullmatches an azure container
    name or a $root/$web/$logs alias."""

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
                "jdbc:postgresql://host:5432/db",
                id="jdbc-keeps-clean-path-drops-userinfo-and-query",
            ),
            pytest.param("jdbc:postgresql://h:5432/db1", "jdbc:postgresql://h:5432/db1", id="jdbc-keeps-db1"),
            pytest.param("jdbc:postgresql://h:5432/db2", "jdbc:postgresql://h:5432/db2", id="jdbc-keeps-db2"),
            pytest.param(
                "'jdbc:postgresql://h:5432/db1'", "'jdbc:postgresql://h:5432/db1'", id="jdbc-quoted-clean-path-kept"
            ),
            pytest.param("jdbc:hive2://host:10000/a/b", "jdbc:hive2://host:10000/a/b", id="jdbc-keeps-multi-segment"),
            pytest.param("jdbc:postgresql://h/my%20db", "jdbc:postgresql://h/my%20db", id="jdbc-keeps-percent-encoded"),
            pytest.param(
                "jdbc:derby://host:1527/db;user=u;password=hunter2",
                "jdbc:derby://host:1527",
                id="jdbc-derby-drops-semicolon-path",
            ),
            pytest.param("zip://d;%74oken=SECRET", "zip://d", id="host-param-encoded-token-is-cut"),
            pytest.param("s3://b;v=2/x.csv", "s3://b;v=2/x.csv", id="host-param-benign-is-kept"),
            pytest.param("zip://d;%76=2", "zip://d;%76=2", id="host-param-encoded-benign-is-kept"),
            pytest.param(
                "jdbc:postgresql://h/db/hunter2@x:5432/y",
                "jdbc:postgresql://h",
                id="jdbc-drops-path-with-at-sign",
            ),
            pytest.param(
                "jdbc:postgresql://h/db%3Bpassword%3Dhunter2",
                "jdbc:postgresql://h/",
                id="jdbc-encoded-secret-cuts-path",
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
                "jdbc:h2:tcp://host/~/test;USER=sa;PASSWORD=hunter2", "jdbc:h2:tcp://host", id="jdbc-h2-tcp-drops-path"
            ),
            pytest.param("jdbc:h2:tcp://host/~/test", "jdbc:h2:tcp://host/~/test", id="jdbc-h2-tcp-keeps-clean-path"),
            pytest.param(
                "jdbc:mysql:loadbalance://h1:3306/db",
                "jdbc:mysql:loadbalance://h1:3306/db",
                id="jdbc-mysql-loadbalance-keeps-clean-path",
            ),
            pytest.param(
                "jdbc:jtds:sqlserver://h/db;user=u;password=x", "jdbc:jtds:sqlserver://h", id="jdbc-jtds-drops-path"
            ),
            pytest.param("jdbc:a:b:c:d://h/x;k=v", "jdbc:a:b:c:d://h", id="jdbc-four-words-boundary"),
            pytest.param(
                "jdbc:jtds:sqlserver://h/db;user=u;password=x, next",
                "jdbc:jtds:sqlserver://h next",
                id="jdbc-jtds-drops-path-keeps-text-after-whitespace",
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
                "s3://bucket/my%20file.csv",
                id="percent-encoded-space-kept",
            ),
            pytest.param(
                "s3://bucket/2024%2F01/a.parquet",
                "s3://bucket/2024%2F01/a.parquet",
                id="percent-encoded-slash-kept",
            ),
            pytest.param(
                "s3://bucket/dir/my%20file.csv",
                "s3://bucket/dir/my%20file.csv",
                id="percent-encoded-space-in-subdir-kept",
            ),
            pytest.param(
                "https://host/a/b%20c/d.csv",
                "https://host/a/b%20c/d.csv",
                id="mid-path-percent-encoded-space-kept",
            ),
            pytest.param("https://host/a%20b.csv", "https://host/a%20b.csv", id="percent-encoded-space-root-kept"),
            pytest.param("https://host/t%3A00.csv", "https://host/t%3A00.csv", id="percent-encoded-colon-kept"),
            pytest.param(
                "https://acct.blob.core.windows.net/c/year%3D2024/a.parquet",
                "https://acct.blob.core.windows.net/c/year%3D2024/a.parquet",
                id="percent-encoded-equals-hive-partition-kept",
            ),
            pytest.param("https://host/caf%C3%A9.csv", "https://host/caf%C3%A9.csv", id="utf8-escape-kept"),
            pytest.param("https://host/a/b%3Fc/d.csv", "https://host/a/", id="encoded-question-mark-mid-path-cut"),
            pytest.param("https://host/x;pass%77ord=hunter2", "https://host/", id="encoded-keyword-letter-cut"),
            pytest.param(
                "https://proxy/fetch/https%3A%2F%2Falice%3Ahunter2%40inner.host%2Fx",
                "https://proxy/fetch/",
                id="encoded-inner-uri-userinfo-cut",
            ),
            pytest.param("https://host/a/b%ZZc/d.csv", "https://host/a/", id="malformed-percent-escape-cut"),
            pytest.param("https://user:123/PW@host/x", "https://host/x", id="digit-userinfo-slash-password"),
            pytest.param(
                "https://user:123/PW@host:8080/x", "https://host:8080/x", id="digit-userinfo-slash-password-port"
            ),
            pytest.param("https://host:8443/a@b.csv", "https://host:8443/a@b.csv", id="path-at-file-kept"),
            pytest.param("https://host:8443/pkg@1.2.3", "https://host:8443/pkg@1.2.3", id="path-at-version-kept"),
            pytest.param(
                "https://host:8443/image@sha256:abc", "https://host:8443/image@sha256:abc", id="path-at-digest-kept"
            ),
            pytest.param(
                "https://host:8443/image@sha256:abc/x",
                "https://host:8443/image@sha256:abc/x",
                id="path-at-digest-with-tail-kept",
            ),
            pytest.param("https://host:8443/a/b@c/d", "https://host:8443/a/b@c/d", id="path-at-mid-segment-kept"),
            pytest.param("https://user:123/PW@host", "https://user:123/PW@host", id="digit-userinfo-no-path-kept"),
            pytest.param("https://host:8443/a@b/c.csv", "https://b/c.csv", id="ambiguous-path-at-host-shape-dropped"),
            pytest.param(
                "https://host:8443/alice@corp.com/report.csv",
                "https://corp.com/report.csv",
                id="ambiguous-path-email-shape-dropped",
            ),
            pytest.param("https://host:80/pkg@1.2.3/", "https://1.2.3/", id="ambiguous-path-version-slash-dropped"),
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
            pytest.param(
                "https://h/x?password=pw]ter2&x=1",
                "https://h/x",
                id="dropped-query-value-with-close-bracket-leaves-no-fragment",
            ),
            pytest.param(
                "https://h/x?password=pw)ter2",
                "https://h/x",
                id="dropped-query-value-with-close-paren-leaves-no-fragment",
            ),
            pytest.param(
                "https://h/x?password=pw'ter2",
                "https://h/x",
                id="dropped-query-value-with-quote-leaves-no-fragment",
            ),
            pytest.param(
                "https://h/a/db:password=pw]ter2",
                "https://h/a",
                id="key-value-path-value-with-close-bracket-leaves-no-fragment",
            ),
            pytest.param("(https://h/x)", "(https://h/x)", id="bracketed-clean-url-keeps-its-closing-paren"),
            pytest.param("'https://h/x'", "'https://h/x'", id="quoted-clean-url-keeps-its-closing-quote"),
            pytest.param(
                "alice:p@ss9://host/db",
                "alice:***://host/db",
                id="password-in-scheme-position-is-masked",
            ),
            pytest.param(
                "user:pw:x://host",
                "user:***://host",
                id="password-and-extra-colon-in-scheme-position-are-masked",
            ),
            pytest.param(
                "x/alice:hunter2://host",
                "x/alice:***://host",
                id="password-in-scheme-position-after-a-slash-is-masked",
            ),
            pytest.param(
                "postgresql://u:12/ss@host",
                "postgresql://host",
                id="slash-in-userinfo-after-digit-segment-is-dropped",
            ),
            pytest.param(
                "{'url': 'https://h/x?a=1', 'password': 'sec'}",
                "{'url': 'https://h/x', 'password': '***'}",
                id="py-dict-url-then-password-keeps-closing-quote",
            ),
            pytest.param(
                '{"url": "https://h/x?a=1", "password": "sec"}',
                '{"url": "https://h/x", "password": "***"}',
                id="json-dict-url-then-password-keeps-closing-quote",
            ),
            pytest.param(
                "{'a': 'see https://h/x?a=1', 'password': 'sec'}",
                "{'a': 'see https://h/x', 'password': '***'}",
                id="py-dict-prose-url-then-password",
            ),
            pytest.param(
                "{'url': 'https://h/x?password=hun]ter2', 'password': 'sec'}",
                "{'url': 'https://h/x', 'password': '***'}",
                id="py-dict-url-secret-bracket-then-password",
            ),
            pytest.param(
                "{'url': 'jdbc:postgresql://h/db;a=1', 'password': 'sec'}",
                "{'url': 'jdbc:postgresql://h', 'password': '***'}",
                id="py-dict-jdbc-url-then-password",
            ),
            pytest.param(
                "{'url':'https://h/x?a=1','password':'sec'}",
                "{'url':'https://h/x','password': '***'}",
                id="py-dict-compact-url-then-password",
            ),
            pytest.param(
                '["https://h1/p?a=1","https://h2/p?b=2","https://h3/p?c=3"]',
                '["https://h1/p","https://h2/p","https://h3/p"]',
                id="compact-json-url-list",
            ),
            pytest.param(
                "[https://h1/p?a=1],[https://h2/p?b=2]",
                "[https://h1/p],[https://h2/p]",
                id="bracketed-urls-comma-separated",
            ),
            pytest.param(
                "{'url':'https://h/x?a=1','table':'sales','err':'timeout'}",
                "{'url':'https://h/x','table':'sales','err':'timeout'}",
                id="py-dict-compact-url-keeps-siblings",
            ),
            pytest.param(
                '{"https://h/x?a=1":{"status":500,"reason":"timeout"}}',
                '{"https://h/x":{"status":500,"reason":"timeout"}}',
                id="json-url-as-key-keeps-nested-value",
            ),
            pytest.param(
                "failed to read 'https://h/x?a=1'. Retrying",
                "failed to read 'https://h/x'. Retrying",
                id="quoted-url-then-period",
            ),
            pytest.param("'https://h/x?a=1': 500", "'https://h/x': 500", id="quoted-url-then-colon"),
            pytest.param("url='https://h/x?a=1'; retry", "url='https://h/x'; retry", id="quoted-url-then-semicolon"),
            pytest.param("(https://h/x?a=1) next", "(https://h/x) next", id="paren-url-then-text"),
            pytest.param("'https://h/x?a=1', next", "'https://h/x', next", id="quoted-url-then-comma"),
            pytest.param("[https://h/x?a=1], next", "[https://h/x], next", id="bracket-url-then-comma"),
            pytest.param("'https://h/x?a=1'", "'https://h/x'", id="quoted-url-with-query-alone"),
            pytest.param("https://h/x?password=hun]ter2", "https://h/x", id="dropped-value-cut-at-close-bracket"),
            pytest.param("https://h/x?password=hun)ter2", "https://h/x", id="dropped-value-cut-at-close-paren"),
            pytest.param("https://h/x?password=hun<ter2", "https://h/x", id="dropped-value-cut-at-lt"),
            pytest.param("https://h/x?password=hun>ter2", "https://h/x", id="dropped-value-cut-at-gt"),
            pytest.param("https://h/x?password=hun'ter2", "https://h/x", id="dropped-value-cut-at-quote"),
            pytest.param('https://h/x?password=hun"ter2', "https://h/x", id="dropped-value-cut-at-double-quote"),
            pytest.param("https://h/x?hun]ter2", "https://h/x", id="bare-query-cut-at-close-bracket"),
            pytest.param("https://h/x?tok)en9", "https://h/x", id="bare-query-cut-at-close-paren"),
            pytest.param("https://h/x#hun]ter2", "https://h/x", id="fragment-cut-at-close-bracket"),
            pytest.param(
                "https://h/x?%70assword%3Dhun]ter2", "https://h/x", id="encoded-keyword-query-cut-at-close-bracket"
            ),
            pytest.param(
                "https://h/x?%2570assword%253Dhun]ter2",
                "https://h/x",
                id="double-encoded-keyword-query-cut-at-close-bracket",
            ),
            pytest.param("zip://d;%74oken=SEC'RET", "zip://d", id="host-param-encoded-token-cut-at-quote"),
            pytest.param("s3://b/%74oken=SEC]RET/x.csv", "s3://b/", id="path-encoded-token-cut-at-close-bracket"),
            pytest.param(
                "zip://d;%74oken=SEC]RET::s3://b/x.csv",
                "zip://d://b/x.csv",
                id="host-param-encoded-token-cut-at-bracket-in-chain",
            ),
            pytest.param(
                "failed: zip://d;%74oken=SECRET::s3://b/x.csv",
                "failed: zip://d::s3://b/x.csv",
                id="host-param-encoded-token-chain-keeps-tail-without-terminator",
            ),
            pytest.param("https://host:8443/pkg@2", "https://host:8443/pkg@2", id="digit-userinfo-path-at-digit-kept"),
            pytest.param(
                "https://cdn.example.com:8443/lodash@4",
                "https://cdn.example.com:8443/lodash@4",
                id="port-path-at-digit-kept",
            ),
            pytest.param("https://reg:5000/img@latest", "https://reg:5000/img@latest", id="port-path-at-word-kept"),
            pytest.param("s3://b:9000/k@v2", "s3://b:9000/k@v2", id="s3-port-path-at-version-kept"),
            pytest.param(
                "postgresql://u:12/ss@db.internal", "postgresql://db.internal", id="db-scheme-digit-slash-dotted-host"
            ),
            pytest.param(
                "postgresql://u:12/ss@db.internal:5432",
                "postgresql://db.internal:5432",
                id="db-scheme-digit-slash-dotted-host-port",
            ),
            pytest.param(
                "postgresql://u:12/ss@host:5432", "postgresql://host:5432", id="db-scheme-digit-slash-host-port"
            ),
            pytest.param("postgresql://u:12/s/s@host", "postgresql://host", id="db-scheme-digit-two-slashes-bare-host"),
            pytest.param(
                "postgresql://u:12/s/s@db.internal",
                "postgresql://db.internal",
                id="db-scheme-digit-two-slashes-dotted-host",
            ),
            pytest.param(
                "postgresql://u:12/s/s@host/db", "postgresql://host/db", id="db-scheme-digit-two-slashes-host-path"
            ),
            pytest.param(
                "postgresql://u:12/ss@host?sslmode=require",
                "postgresql://host",
                id="db-scheme-digit-slash-host-query",
            ),
            pytest.param(
                "postgresql+psycopg2://u:12/ss@db.internal",
                "postgresql+psycopg2://db.internal",
                id="db-scheme-plus-suffix-digit-slash",
            ),
            pytest.param(
                "POSTGRESQL://u:12/ss@db.internal", "POSTGRESQL://db.internal", id="db-scheme-upper-case-digit-slash"
            ),
            pytest.param(
                "mongodb+srv://u:12/ss@db.internal", "mongodb+srv://db.internal", id="mongodb-srv-digit-slash"
            ),
            pytest.param(
                "rediss://u:12/ss@cache.internal:6380", "rediss://cache.internal:6380", id="rediss-digit-slash"
            ),
            pytest.param("amqp://u:12/ss@mq.internal", "amqp://mq.internal", id="amqp-digit-slash"),
            pytest.param("clickhouse://u:12/ss@db.internal", "clickhouse://db.internal", id="clickhouse-digit-slash"),
            pytest.param("'postgresql://u:12/ss@host'", "'postgresql://host'", id="db-scheme-digit-slash-quoted"),
            pytest.param(
                "(postgresql://u:12/ss@host)", "(postgresql://host)", id="db-scheme-digit-slash-parenthesised"
            ),
            pytest.param("postgresql://h:5432/db", "postgresql://h:5432/db", id="db-scheme-port-path-kept"),
            pytest.param(
                "postgresql://h:5432/db@prod",
                "postgresql://h:5432/db@prod",
                id="db-scheme-port-path-at-word-source-handle-stays-readable",
            ),
            pytest.param("user:pw:9x://host", "user:***://host", id="scheme-position-password-digit-led-extra-colon"),
            pytest.param("ERROR:root:zip://d/x.csv", "ERROR:root:zip://d/x.csv", id="log-prefix-error-root-kept"),
            pytest.param(
                "WARNING:mloda:hf://datasets/x", "WARNING:mloda:hf://datasets/x", id="log-prefix-warning-mloda-kept"
            ),
            pytest.param("INFO:mloda:rediss://h/0", "INFO:mloda:rediss://h/0", id="log-prefix-info-mloda-kept"),
            pytest.param("ERROR:mloda:memory://t/x", "ERROR:mloda:memory://t/x", id="log-prefix-error-mloda-kept"),
            pytest.param(
                "ERROR:root:https://h/x failed", "ERROR:root:https://h/x failed", id="log-prefix-then-prose-kept"
            ),
            pytest.param("WARNING:mloda:s3://b/x.csv", "WARNING:mloda:s3://b/x.csv", id="log-prefix-s3-kept"),
            pytest.param("url:https://h/x", "url:https://h/x", id="label-colon-url-kept"),
            pytest.param("dsn:postgres://h/db", "dsn:postgres://h/db", id="label-colon-dsn-kept"),
            pytest.param("s3://bucket/key:https://h/x", "s3://bucket/key:https://h/x", id="uri-colon-uri-kept"),
            pytest.param("/var/log/app:https://h/x", "/var/log/app:https://h/x", id="path-colon-url-kept"),
            pytest.param("simplecache::s3://b/x.csv", "simplecache::s3://b/x.csv", id="chain-double-colon-kept"),
            pytest.param("x:jdbc:h2://h/db", "x:jdbc:h2://h/db", id="label-colon-jdbc-kept"),
            pytest.param("a:://h", "a:://h", id="empty-scheme-after-colon-kept"),
            pytest.param("see a:b:c://h", "see a:***://h", id="accepted-masking-two-colon-prefix-treated-as-password"),
            pytest.param(
                "/var/log/app:zip://h/x",
                "/var/log/app:***://h/x",
                id="accepted-masking-path-colon-known-scheme-treated-as-password",
            ),
            pytest.param(
                "postgresql://h:5432/db?password=p@ssw0rd", "postgresql://h:5432/db", id="at-in-query-password-port"
            ),
            pytest.param(
                "postgresql://h:12/db?password=p@ssw0rd", "postgresql://h:12/db", id="at-in-query-password-digit-port"
            ),
            pytest.param(
                "jdbc:mysql://h:3306/db?user=u&password=p@ssword",
                "jdbc:mysql://h:3306/db",
                id="at-in-jdbc-query-password",
            ),
            pytest.param(
                "postgresql://h:5432/db;password=hun@ter2",
                "postgresql://h:5432/db;password=***",
                id="at-in-semicolon-password-masked-in-place",
            ),
            pytest.param(
                "jdbc:postgresql://h:5432/db;password=hun@ter2",
                "jdbc:postgresql://h:5432",
                id="at-in-jdbc-semicolon-password-drops-path",
            ),
            pytest.param(
                "postgresql://h:5432/db?user=alice@corp&sslcert=SECRET",
                "postgresql://h:5432/db",
                id="at-in-query-user-then-secret-key",
            ),
            pytest.param(
                "mongodb://h:27017/db?authSource=admin@x&sharedkey=SECRET",
                "mongodb://h:27017/db",
                id="at-in-query-auth-source-then-secret-key",
            ),
            pytest.param("redis://h:6379/0?auth=p@ss", "redis://h:6379/0", id="at-in-query-auth-value"),
            pytest.param("ftp://h:21/x#tok@en", "ftp://h:21/x", id="at-in-fragment-token"),
            pytest.param(
                '{"url":"kafka://b:9092/t","email":"a@b.com"}',
                '{"url":"kafka://b:9092/t","email":"a@b.com"}',
                id="email-in-sibling-json-value-kept",
            ),
            pytest.param(
                "postgresql://h:5432/db,admin@corp.com",
                "postgresql://h:5432/db,admin@corp.com",
                id="email-after-comma-kept",
            ),
            pytest.param(
                "postgresql://db.internal:5432/sales@prod",
                "postgresql://db.internal:5432/sales@prod",
                id="source-handle-dotted-host-kept",
            ),
            pytest.param(
                "Two sources: postgresql://db.internal:5432/sales@prod, postgresql://db.internal:5432/sales@staging",
                "Two sources: postgresql://db.internal:5432/sales@prod, postgresql://db.internal:5432/sales@staging",
                id="source-handles-two-in-prose-kept",
            ),
            pytest.param(
                "mysql://h:3306/db@replica matched but table gone",
                "mysql://h:3306/db@replica matched but table gone",
                id="source-handle-with-prose-tail-kept",
            ),
            pytest.param("sftp://h:22/data/x.csv@a", "sftp://h:22/data/x.csv@a", id="path-at-sftp-file-kept"),
            pytest.param("sftp://host:22/home/u@dom/f", "sftp://host:22/home/u@dom/f", id="path-at-sftp-mid-kept"),
            pytest.param("kafka://b:9092/t@v", "kafka://b:9092/t@v", id="path-at-kafka-topic-version-kept"),
            pytest.param("ssh://host:22/repo@main", "ssh://host:22/repo@main", id="path-at-ssh-ref-kept"),
            pytest.param("ftp://host:21/file@v1.tar", "ftp://host:21/file@v1.tar", id="path-at-ftp-file-version-kept"),
            pytest.param(
                "docker://reg:5000/img@sha256:abc", "docker://reg:5000/img@sha256:abc", id="path-at-docker-digest-kept"
            ),
            pytest.param("hf://h:9/ds@main", "hf://h:9/ds@main", id="path-at-hf-dataset-ref-kept"),
            pytest.param("zip://h:1/a@b", "zip://h:1/a@b", id="path-at-zip-single-digit-port-kept"),
            pytest.param("iceberg://h:80/ns@v", "iceberg://h:80/ns@v", id="path-at-iceberg-namespace-version-kept"),
            pytest.param(
                "postgres://user:pw@db:5432/x@y", "postgres://db:5432/x@y", id="real-userinfo-then-path-at-kept"
            ),
            pytest.param("mysql://u:p@host:3306/db@v", "mysql://host:3306/db@v", id="real-userinfo-then-db-at-kept"),
            pytest.param("redis://:pw@h:6379/0@x", "redis://h:6379/0@x", id="empty-user-real-userinfo-then-at-kept"),
            pytest.param(
                "sftp://u:pw@host:22/home/a@b", "sftp://host:22/home/a@b", id="sftp-real-userinfo-then-path-at-kept"
            ),
            pytest.param("redis://:12/ss@host", "redis://host", id="empty-user-digit-slash-redis"),
            pytest.param("postgresql://:12/ss@host", "postgresql://host", id="empty-user-digit-slash-postgresql"),
            pytest.param("mysql://:9/x/y@h", "mysql://h", id="empty-user-digit-two-slashes-mysql"),
            pytest.param(
                "postgresql://u:5432/ss@db.internal", "postgresql://db.internal", id="plausible-port-slash-dotted-host"
            ),
            pytest.param(
                "postgresql://u:5432/ss@db.internal:5432",
                "postgresql://db.internal:5432",
                id="plausible-port-slash-dotted-host-port",
            ),
            pytest.param(
                "postgresql://u:5432/ss@host:5432", "postgresql://host:5432", id="plausible-port-slash-host-port"
            ),
            pytest.param(
                "jdbc:postgresql://h/db;a=pw],'ter2", "jdbc:postgresql://h", id="jdbc-value-closer-comma-quote-cut"
            ),
            pytest.param(
                "jdbc:postgresql://h/db;password=pw]:[ter2",
                "jdbc:postgresql://h",
                id="jdbc-value-closer-colon-opener-cut",
            ),
            pytest.param(
                "jdbc:sqlserver://h;password=pw],'ter2",
                "jdbc:sqlserver://h",
                id="jdbc-host-param-closer-comma-quote-cut",
            ),
            pytest.param("https://h/x?password=hun],'ter2", "https://h/x", id="query-value-closer-comma-quote-cut"),
            pytest.param("https://h/x?password=hun):{ter2", "https://h/x", id="query-value-closer-colon-brace-cut"),
            pytest.param("https://h/x?password=a],'b", "https://h/x", id="query-short-value-closer-comma-quote-cut"),
            pytest.param("https://h/x?password=a)):'b", "https://h/x", id="query-value-double-closer-colon-quote-cut"),
            pytest.param(
                "https://a;jdbc:postgresql://h/db;user=x;mypass=y",
                "https://a;jdbc:postgresql://h",
                id="semicolon-before-jdbc-uri-is-not-host-param",
            ),
            pytest.param(
                "https://a;jdbc:postgresql://h:1/db;sslkey=y",
                "https://a;jdbc:postgresql://h:1",
                id="semicolon-before-jdbc-uri-with-port-is-not-host-param",
            ),
            pytest.param("zip://d%3Btoken%3DSECRET", "zip://d", id="host-encoded-semicolon-and-equals-cut"),
            pytest.param("zip://d%3B%74oken=SECRET", "zip://d", id="host-encoded-semicolon-encoded-letter-cut"),
            pytest.param("zip://%74oken=SECRET", "zip://", id="empty-host-encoded-letter-key-cut"),
            pytest.param("zip://%74oken%3DSECRET", "zip://", id="empty-host-encoded-letter-and-equals-cut"),
            pytest.param("zip://d%74oken=SECRET", "zip://d", id="host-encoded-letter-key-without-semicolon-cut"),
            pytest.param("zip://d:80%3Btoken%3DSECRET", "zip://d:80", id="host-port-encoded-semicolon-cut"),
            pytest.param(
                "zip://d%3Btoken%3DSECRET::s3://b/x.csv",
                "zip://d::s3://b/x.csv",
                id="host-encoded-semicolon-cut-keeps-chain-tail",
            ),
            pytest.param("s3://b%20x/y", "s3://b%20x/y", id="host-encoded-space-kept"),
            pytest.param("s3://b%2Fc/y", "s3://b%2Fc/y", id="host-encoded-slash-kept"),
            pytest.param("zip://d%41::s3://b/x", "zip://d%41::s3://b/x", id="host-encoded-letter-benign-chain-kept"),
            pytest.param("error:root:zip://d/x.csv", "error:root:zip://d/x.csv", id="log-prefix-lower-case-error-kept"),
            pytest.param("TRACE:root:zip://d/x.csv", "TRACE:root:zip://d/x.csv", id="log-prefix-trace-kept"),
            pytest.param("NOTICE:root:hf://d/x", "NOTICE:root:hf://d/x", id="log-prefix-notice-kept"),
            pytest.param("alice:jdbc:x://host", "alice:jdbc:x://host", id="label-colon-jdbc-subprotocol-kept"),
            pytest.param("alice:jdbc://host", "alice:***://host", id="password-before-bare-jdbc-scheme-masked"),
            pytest.param("alice:JDBC://host", "alice:***://host", id="password-before-upper-case-jdbc-scheme-masked"),
        ],
    )
    def test_scrub_keeps_host_and_path_only(self, uri: str, expected: str) -> None:
        assert scrub_credentials(uri) == expected

    def test_long_uri_without_secrets_returns_quickly(self) -> None:
        value = "s3://b/" + "a" * 40000
        assert scrub_credentials(value) == value

    def test_long_uri_with_a_trailing_invalid_character_returns_quickly(self) -> None:
        _assert_scrubs_linearly(
            lambda k: "a://" + "a@" * k + "/" + "a" * k + "!", 20000, "a long uri with a trailing invalid character"
        )


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
            pytest.param("https://host/x;api%5Fkey=SECRET", "SECRET", id="encoded-underscore-in-secret-key"),
            pytest.param("https://host/db%3Ajob=SECRETVAL", "SECRETVAL", id="encoded-colon-then-equals-in-path"),
            pytest.param(
                "https://host/db%3Fpassword%3Dhunter2/x:y=z", "hunter2", id="encoded-secret-before-fallback-pair"
            ),
            pytest.param("https://host/r/s3%3A%2F%2Ftoken%40bucket/x", "token", id="encoded-userinfo-in-path"),
            pytest.param(
                "https://proxy/f/https%3A%2F%2Falice%3A123%2Fpw9secret%40inner",  # nosec B105
                "pw9secret",
                id="encoded-digit-userinfo-slash-password",
            ),
            pytest.param("https://host/x/scott%2Ftiger%40db%3A1521", "tiger", id="encoded-slash-password-at-host"),
            pytest.param(
                "https://host/a/user%2Fhunter2%40db.example.com:5432/x", "hunter2", id="encoded-slash-encoded-at"
            ),
            pytest.param("https://host/a/user%2Fhunter2@db.example.com:5432/x", "hunter2", id="encoded-slash-raw-at"),
            pytest.param("https://h/x%22password%22%3A%22hunter2%22", "hunter2", id="encoded-json-pair-in-path"),
            pytest.param("jdbc:postgresql://h/db;password=hun]ter2", "ter2", id="jdbc-close-bracket-in-secret"),
            pytest.param("jdbc:postgresql://h/db;password=hun>ter2", "ter2", id="jdbc-gt-in-secret"),
            pytest.param("jdbc:h2:tcp://host/~/t;PASSWORD=hun]ter2", "ter2", id="jdbc-h2-close-bracket-in-secret"),
            pytest.param(
                "jdbc:postgresql://h/scott/tiger'@host:1521/svc", "tiger", id="jdbc-quote-before-at-after-path"
            ),
            pytest.param(
                'jdbc:postgresql://h/scott/tiger"@host:1521/svc', "tiger", id="jdbc-double-quote-before-at-after-path"
            ),
            pytest.param("https://host/x.csv;PHPSESSID=abc123secret", "abc123secret", id="path-param-phpsessid"),
            pytest.param("https://host/x.csv;token=abc123secret", "abc123secret", id="path-param-token"),
            pytest.param("https://host/x.csv;API-KEY=abc123secret", "abc123secret", id="path-param-api-key"),
            pytest.param(
                "https://host/x.csv;x-amz-signature=abc123secret", "abc123secret", id="path-param-amz-signature"
            ),
            pytest.param("https://host/x.csv;sessid=abc123secret", "abc123secret", id="path-param-sessid"),
            pytest.param("s3://b/d;aws.token=SECRET/x.csv", "SECRET", id="path-param-dotted-key"),
            pytest.param("s3://b/d;x~token=SECRET/x.csv", "SECRET", id="path-param-tilde-key"),
            pytest.param("s3://b/d;v=1,token=SECRETCOMMA/x.csv", "SECRETCOMMA", id="path-param-comma-separated"),
            pytest.param("s3://b/d;v=1&token=SECRETAMP/x.csv", "SECRETAMP", id="path-param-ampersand"),
            pytest.param("s3://b/d;token = SECRETSPACE/x.csv", "SECRETSPACE", id="path-param-spaces-around-equals"),
            pytest.param("s3://b/d;token=SECRETVAL,b/x.csv", "SECRETVAL", id="path-param-comma-in-value"),
            pytest.param("postgresql://u:12/ss@host/db", "ss", id="digit-userinfo-slash-password"),
            pytest.param("https://tok/en9x@host/x", "en9x", id="slash-in-userinfo-token"),
            pytest.param("https://tok/en9x@host", "en9x", id="slash-in-userinfo-token-bare-host"),
            pytest.param("(https://tok/en9x@host)", "en9x", id="slash-in-userinfo-token-parenthesised"),
            pytest.param("https://tok/en9x@host/", "en9x", id="slash-in-userinfo-token-trailing-slash"),
            pytest.param("https://tok/en9x@host?x=1", "en9x", id="slash-in-userinfo-token-query"),
            pytest.param("alice:hunter2://u:pw9x@host/db", "pw9x", id="scheme-position-then-userinfo-pw"),
            pytest.param("alice:hunter2://u:pw9x@host/db", "hunter2", id="scheme-position-then-userinfo-name"),
            pytest.param("alice:9hunter://host/db", "9hunter", id="scheme-position-digit-led"),
            pytest.param("user:p_w!x://h/db", "p_w!x", id="scheme-position-punctuation"),
            pytest.param("https://h/x?password=pw]ter2&x=1", "ter2", id="query-close-bracket-in-secret"),
            pytest.param("https://h/x?password=pw)ter2&x=1", "ter2", id="query-close-paren-in-secret"),
            pytest.param("https://h/x?password=pw<ter2&x=1", "ter2", id="query-lt-in-secret"),
            pytest.param("https://h/x?password=pw>ter2&x=1", "ter2", id="query-gt-in-secret"),
            pytest.param("https://h/x?password=pw'ter2&x=1", "ter2", id="query-quote-in-secret"),
            pytest.param('https://h/x?password=pw"ter2&x=1', "ter2", id="query-double-quote-in-secret"),
            pytest.param("https://h/a/db:password=pw]ter2", "ter2", id="key-value-path-close-bracket-in-secret"),
            pytest.param("zip://d;%74oken=SECRETVAL", "SECRETVAL", id="host-param-encoded-token"),
            pytest.param(
                "failed: zip://d;%74oken=SECRETVAL::s3://b/x.csv", "SECRETVAL", id="host-param-encoded-token-in-chain"
            ),
            pytest.param("https://host;%70assword=SECRETVAL", "SECRETVAL", id="host-param-encoded-password"),
            pytest.param("zip://d;api%5Fkey=SECRETVAL", "SECRETVAL", id="host-param-encoded-underscore-key"),
            pytest.param("zip://d;%74oken=SECRETVAL/path", "SECRETVAL", id="host-param-encoded-token-before-path"),
            pytest.param("alice:p@ss9://host/db", "p@ss9", id="at-sign-in-password-in-scheme-position"),
            pytest.param("user:pw:x://host", "pw", id="extra-colon-after-password-in-scheme-position"),
            pytest.param("x/alice:hunter2://host", "hunter2", id="slash-before-user-in-scheme-position"),
            pytest.param("postgresql://u:12/ss@host", "12/ss", id="slash-in-userinfo-after-digit-segment-no-path"),
            pytest.param(
                '{"url":"https://h/x?a=1","password":"s://ec"}', "s://ec", id="json-compact-password-with-scheme-shape"
            ),
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

    def test_data_access_collection_redact_masks_a_secret_after_an_anchor_literal(self) -> None:
        from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection

        dac = DataAccessCollection(credentials={"h1": {"flavor": "postgresql", "token": _LITERAL}})

        assert "pw7" not in dac.redact("postgresql+psycopg2://alice:pw7@db/x")

    @pytest.mark.parametrize(
        ("literal", "text", "secret"),
        [
            pytest.param("postgresql", "failed: postgresql+psycopg2://alice:pw7@db/x", "pw7", id="anchor-scheme"),
            pytest.param("Bearer", "header Bearer abc123.tok", "abc123.tok", id="anchor-bearer"),
            pytest.param("password", "conn password=pw1 host", "pw1", id="anchor-password"),
            pytest.param("Tr0ub4dor&3xyz", "password=Tr0ub4dor&3xyz", "3xyz", id="ampersand-in-literal"),
            pytest.param("abc;def123456", "password=abc;def123456 end", "def123456", id="semicolon-in-literal"),
            pytest.param("pass word 12345", "password=pass word 12345", "word 12345", id="space-in-literal"),
            pytest.param("abcd1234;efgh5678", "Bearer abcd1234;efgh5678", "efgh5678", id="semicolon-after-bearer"),
            pytest.param("password", "password=xpasswordy password=pw1", "pw1", id="password-inside-a-value"),
            pytest.param(
                "Bearer",
                "Authorization: Bearer abc123 Bearer zz; then Bearer tok9xyz",
                "tok9xyz",
                id="bearer-repeated-later-token",
            ),
            pytest.param(
                "Bearer",
                "Authorization: Bearer abc123 Bearer zz; then Bearer tok9xyz",
                "abc123",
                id="bearer-repeated-first-token",
            ),
            pytest.param(
                "Bearer",
                "header Bearer abc123.tok access_token=Bearer9",
                "abc123.tok",
                id="bearer-then-access-token",
            ),
            pytest.param(
                "postgresql",
                "postgresql://u:postgresql@h/x and postgresql+psycopg2://alice:pw7@db/x",
                "pw7",
                id="anchor-scheme-after-literal-password",
            ),
            pytest.param(
                "postgresql://alice", "postgresql://alice:pw9x@db/x", "pw9x", id="anchor-scheme-with-userinfo-name"
            ),
            pytest.param(
                "abc;def123456", "https://host/abc;def123456/x.csv", "def123456", id="semicolon-literal-in-url-path"
            ),
            pytest.param(
                "hun]ter2",
                "{'url': 'https://h/x?password=hun]ter2', 'password': 'sec'}",
                "hun",
                id="bracket-literal-in-query-head-gone",
            ),
            pytest.param(
                "hun]ter2",
                "{'url': 'https://h/x?password=hun]ter2', 'password': 'sec'}",
                "ter2",
                id="bracket-literal-in-query-tail-gone",
            ),
            pytest.param(
                "hun]ter2",
                "{'url': 'https://h/x?password=hun]ter2', 'password': 'sec'}",
                "sec",
                id="bracket-literal-in-query-sibling-password-gone",
            ),
            pytest.param("x'y", "'https://h/x?a=1x'y'", "x'y", id="quote-literal-after-query"),
        ],
    )
    def test_secret_is_masked_when_the_literal_overlaps_a_pattern(self, literal: str, text: str, secret: str) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

        with registered_secrets_scope(frozenset({literal})):
            masked = scrub_credentials(text)

        assert secret not in masked
        assert literal not in masked

    def test_whole_dsn_literal_is_masked_verbatim_and_re_rendered(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

        literal = "postgresql://alice:LongPassw0rd@db.example.com/x"  # nosec B105
        with registered_secrets_scope(frozenset({literal})):
            verbatim = scrub_credentials(f"verbatim {literal}")
            rendered = scrub_credentials("postgresql+psycopg2://alice:LongPassw0rd@db.example.com/x")  # nosec B105

        assert "LongPassw0rd" not in verbatim
        assert "alice" not in verbatim
        assert "LongPassw0rd" not in rendered
        assert "alice:" not in rendered

    @pytest.mark.parametrize(
        ("literal", "text"),
        [
            ("hunter2!xyz", "https://host/t/hunter2%21xyz"),  # nosec B105
            ("abc/def+ghi=", "https://host/files/abc%2Fdef%2Bghi%3D.csv"),  # nosec B105
            ("s3cret pass word", "https://host/f/s3cret%20pass%20word/x"),  # nosec B105
        ],
    )
    def test_percent_encoded_literal_in_a_uri_path_is_masked(self, literal: str, text: str) -> None:
        from urllib.parse import quote

        from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

        with registered_secrets_scope(frozenset({literal})):
            masked = scrub_credentials(text)

        assert literal not in masked
        assert quote(literal, safe="") not in masked


def test_mask_literals_masks_every_occurrence_and_merges_overlaps() -> None:
    from mloda.core.abstract_plugins.components.credential_scrub import mask_literals

    masked = mask_literals(f"a {_LITERAL} b {_LITERAL}", {_LITERAL})
    assert _LITERAL not in masked
    assert masked.startswith("a ")
    assert mask_literals("nothing here", {_LITERAL}) == "nothing here"


class _Pair(NamedTuple):
    first: str
    second: str


class _KeepList(list[Any]):
    pass


class _CodedGroup(ExceptionGroup):
    """A group whose __new__ takes a code, so a rebuild must not call it."""

    code: int

    def __new__(cls, message: str, excs: Any, code: int) -> _CodedGroup:
        self = super().__new__(cls, message, excs)
        self.code = code
        self.args = (message, excs, code)
        return self


@dataclass(frozen=True, init=False)
class _FrozenGroup(ExceptionGroup):
    detail: str = "d"


def cast_group(exc: BaseException) -> Any:
    assert isinstance(exc, BaseExceptionGroup)
    return exc


def _group_texts(root: BaseException) -> list[str]:
    """Every message and str arg reachable through members, causes and contexts, walked iteratively."""
    texts: list[str] = []
    seen: set[int] = set()
    pending: list[BaseException] = [root]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, BaseExceptionGroup):
            texts.append(current.message)
            pending.extend(current.exceptions)
        for arg in current.args:
            if isinstance(arg, str):
                texts.append(arg)
            if isinstance(arg, (list, tuple)):
                pending.extend(item for item in arg if isinstance(item, BaseException))
                texts.extend(item for item in arg if isinstance(item, str))
        pending.extend(n for n in (current.__cause__, current.__context__) if n is not None)
    return texts


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

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            pytest.param(_LITERAL, "***", id="str"),
            pytest.param({"k": _LITERAL}, {"k": "***"}, id="dict-value"),
            pytest.param({_LITERAL: "v"}, {"***": "v"}, id="dict-key"),
            pytest.param([_LITERAL, "keep"], ["***", "keep"], id="list"),
            pytest.param((_LITERAL, "keep"), ("***", "keep"), id="tuple"),
            pytest.param(_Pair(_LITERAL, "keep"), _Pair("***", "keep"), id="namedtuple"),
        ],
    )
    def test_an_instance_attribute_is_masked(self, payload: Any, expected: Any) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        class TokenHolderError(Exception):
            def __init__(self, token: Any) -> None:
                super().__init__("failed")
                self.token = token

            def __str__(self) -> str:
                return repr(self.token)

        exc = TokenHolderError(payload)

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in str(exc)
        assert exc.token == expected
        assert type(exc.token) is type(expected)

    def test_non_dict_mapping_attributes_are_left_as_is(self) -> None:
        from types import MappingProxyType

        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        exc = ValueError("failed")
        as_proxy = MappingProxyType({"k": _LITERAL})
        setattr(exc, "as_proxy", as_proxy)

        redact_exception(exc, frozenset({_LITERAL}))

        assert getattr(exc, "as_proxy") is as_proxy
        assert as_proxy["k"] == _LITERAL

    @pytest.mark.parametrize("kind", [set, frozenset])
    def test_a_set_attribute_is_masked_into_a_plain_set_of_the_same_kind(self, kind: type) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        exc = ValueError("failed")
        setattr(exc, "as_set", kind({_LITERAL, "keep"}))

        redact_exception(exc, frozenset({_LITERAL}))

        assert type(getattr(exc, "as_set")) is kind
        assert getattr(exc, "as_set") == kind({"***", "keep"})

    @pytest.mark.parametrize("kind", [set, frozenset])
    def test_a_set_attribute_holding_no_literal_keeps_its_identity(self, kind: type) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        exc = ValueError("failed")
        untouched = kind({"keep", "other"})
        setattr(exc, "as_set", untouched)

        redact_exception(exc, frozenset({_LITERAL}))

        assert getattr(exc, "as_set") is untouched

    @pytest.mark.parametrize("kind", [set, frozenset])
    def test_mask_item_masks_a_set_and_keeps_an_unchanged_one(self, kind: type) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import _mask_item

        unchanged = kind({"a", "b"})

        assert _mask_item(unchanged, frozenset({_LITERAL})) is unchanged
        assert _mask_item(kind({_LITERAL, "a"}), frozenset({_LITERAL})) == kind({"***", "a"})

        collapsed = _mask_item(kind({_LITERAL, _OTHER_LITERAL}), frozenset({_LITERAL, _OTHER_LITERAL}))
        assert type(collapsed) is kind
        assert collapsed == kind({"***"})

    def test_a_set_nested_in_an_arg_is_masked(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        exc = ValueError("bad", {_LITERAL, "keep"})

        redact_exception(exc, frozenset({_LITERAL}))

        assert exc.args == ("bad", {"***", "keep"})

    @pytest.mark.parametrize("form", list(_FORM_BUILDERS))
    def test_every_form_of_a_registered_literal_is_masked(self, form: str) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        echoed = _FORM_BUILDERS[form](_FORM_LITERAL)
        exc = ValueError(f"x {echoed} y")
        setattr(exc, "detail", f"d {echoed}")

        redact_exception(exc, frozenset({_FORM_LITERAL}))

        assert str(exc) == "x *** y"
        assert getattr(exc, "detail") == "d ***"

    @pytest.mark.parametrize(
        ("factory", "expected"),
        [
            pytest.param(
                lambda: PermissionError("prod_db may not be used in this run"),
                "prod_db may not be used in this run",
                id="permission-one-arg",
            ),
            pytest.param(lambda: ConnectionError("x"), "x", id="connection"),
            pytest.param(lambda: TimeoutError("t"), "t", id="timeout"),
            pytest.param(lambda: FileNotFoundError("gone"), "gone", id="file-not-found-one-arg"),
            pytest.param(lambda: OSError(2), "2", id="errno-only"),
            pytest.param(
                lambda: FileNotFoundError(2, "No such file", "/data/x"),
                "[Errno 2] No such file: '/data/x'",
                id="errno-strerror-filename",
            ),
            pytest.param(
                lambda: OSError(2, "No such file", "/data/x", None, "/data/y"),
                "[Errno 2] No such file: '/data/x' -> '/data/y'",
                id="five-arg-filename2",
            ),
        ],
    )
    def test_an_oserror_not_holding_the_literal_keeps_its_exact_text(
        self, factory: Callable[[], BaseException], expected: str
    ) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        exc = factory()
        args = exc.args
        assert str(exc) == expected

        redact_exception(exc, frozenset({_LITERAL}))

        assert str(exc) == expected
        assert exc.args == args

    @pytest.mark.parametrize(
        ("factory", "expected"),
        [
            pytest.param(
                lambda: OSError(2, f"bad {_LITERAL}", "/data/x"), "[Errno 2] bad ***: '/data/x'", id="strerror"
            ),
            pytest.param(
                lambda: OSError(2, "No such file", f"/x/{_LITERAL}"),
                "[Errno 2] No such file: '/x/***'",
                id="filename",
            ),
            pytest.param(
                lambda: OSError(2, "No such file", "/data/x", None, f"/y/{_LITERAL}"),
                "[Errno 2] No such file: '/data/x' -> '/y/***'",
                id="filename2",
            ),
        ],
    )
    def test_an_oserror_holding_the_literal_masks_only_that_part_of_its_text(
        self, factory: Callable[[], BaseException], expected: str
    ) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        exc = factory()

        redact_exception(exc, frozenset({_LITERAL}))

        assert str(exc) == expected

    @pytest.mark.parametrize(
        ("factory", "field", "expected"),
        [
            pytest.param(lambda: OSError(2, f"s {_LITERAL}"), "strerror", "s ***", id="oserror-strerror"),
            pytest.param(lambda: OSError(2, "s", f"/f/{_LITERAL}"), "filename", "/f/***", id="oserror-filename"),
            pytest.param(
                lambda: OSError(2, "s", "/a", None, f"/f/{_LITERAL}"), "filename2", "/f/***", id="oserror-filename2"
            ),
            pytest.param(lambda: ImportError(f"m {_LITERAL}"), "msg", "m ***", id="import-msg"),
            pytest.param(lambda: ImportError("m", name=f"n.{_LITERAL}"), "name", "n.***", id="import-name"),
            pytest.param(lambda: ImportError("m", path=f"/p/{_LITERAL}"), "path", "/p/***", id="import-path"),
            pytest.param(lambda: SyntaxError(f"m {_LITERAL}"), "msg", "m ***", id="syntax-msg"),
            pytest.param(
                lambda: SyntaxError("m", (f"/f/{_LITERAL}", 1, 1, "t")), "filename", "/f/***", id="syntax-filename"
            ),
            pytest.param(lambda: SyntaxError("m", ("f", 1, 1, f"t {_LITERAL}")), "text", "t ***", id="syntax-text"),
            pytest.param(
                lambda: UnicodeEncodeError("utf-8", f"x{_LITERAL}", 0, 1, "r"), "object", "x***", id="encode-object"
            ),
            pytest.param(
                lambda: UnicodeEncodeError("utf-8", "x", 0, 1, f"r {_LITERAL}"), "reason", "r ***", id="encode-reason"
            ),
            pytest.param(
                lambda: UnicodeDecodeError("utf-8", b"\xff", 0, 1, f"r {_LITERAL}"),
                "reason",
                "r ***",
                id="decode-reason",
            ),
            pytest.param(
                lambda: UnicodeTranslateError(f"x{_LITERAL}", 0, 1, "r"), "object", "x***", id="translate-object"
            ),
            pytest.param(
                lambda: UnicodeTranslateError("x", 0, 1, f"r {_LITERAL}"), "reason", "r ***", id="translate-reason"
            ),
            pytest.param(lambda: SystemExit(f"bye {_LITERAL}"), "code", "bye ***", id="system-exit-code"),
        ],
    )
    def test_a_native_str_field_is_masked(
        self, factory: Callable[[], BaseException], field: str, expected: str
    ) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        exc = factory()

        redact_exception(exc, frozenset({_LITERAL}))

        assert getattr(exc, field) == expected
        assert _LITERAL not in str(exc)

    def test_a_subclass_property_cannot_shadow_or_reject_a_native_field(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        class ShadowImportError(ImportError):
            @property
            def msg(self) -> str:  # type: ignore[override]
                return "shadow"

        class ShadowOSError(OSError):
            @property
            def strerror(self) -> str:  # type: ignore[override]
                return "shadow"

        imp = ShadowImportError(f"m {_LITERAL}")
        err = ShadowOSError(2, f"s {_LITERAL}")

        redact_exception(imp, frozenset({_LITERAL}))
        redact_exception(err, frozenset({_LITERAL}))

        assert vars(ImportError)["msg"].__get__(imp) == "m ***"
        assert vars(OSError)["strerror"].__get__(err) == "s ***"

    def test_a_non_str_native_or_unlisted_field_keeps_its_identity(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        obj_value = [_LITERAL]
        stop_value = [_LITERAL]
        raw_name = _LITERAL.encode()
        attribute = AttributeError("x", name="n", obj=obj_value)
        stop = StopIteration(stop_value)
        os_error = OSError(2, "s", raw_name)
        exit_code = SystemExit(7)

        for exc in (attribute, stop, os_error, exit_code):
            redact_exception(exc, frozenset({_LITERAL}))

        assert attribute.obj is obj_value
        assert obj_value == [_LITERAL]
        assert stop.value is stop_value
        assert os_error.filename is raw_name
        assert exit_code.code == 7

    def test_a_deeply_nested_list_attribute_is_masked_without_recursion_error(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        nested: list[Any] = [_LITERAL]
        for _ in range(5000):
            nested = [nested]
        exc = ValueError("failed")
        setattr(exc, "deep", nested)

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in repr(getattr(exc, "deep"))

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            pytest.param(f"d {_LITERAL}", "d ***", id="str-field"),
            pytest.param({"k": _LITERAL}, {"k": "***"}, id="dict-field"),
        ],
    )
    def test_a_frozen_dataclass_exception_repr_is_masked(self, payload: Any, expected: Any) -> None:
        from dataclasses import dataclass

        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        @dataclass(frozen=True)
        class FrozenDetailError(Exception):
            detail: Any

        exc = FrozenDetailError(payload)

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in repr(exc)
        assert _LITERAL not in str(exc)
        assert exc.detail == expected

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(lambda: OrderedDict(a="keep"), id="ordereddict"),
            pytest.param(lambda: defaultdict(list, {"a": ["keep"]}), id="defaultdict"),
            pytest.param(lambda: _KeepList(["keep"]), id="list-subclass"),
            pytest.param(lambda: os.stat("."), id="stat-result"),
            pytest.param(lambda: time.gmtime(0), id="struct-time"),
            pytest.param(lambda: ["keep", 1], id="plain-list"),
        ],
    )
    def test_a_container_without_a_literal_keeps_identity_and_type(self, factory: Callable[[], Any]) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        value = factory()
        exc = ValueError("failed")
        setattr(exc, "held", value)

        redact_exception(exc, frozenset({_LITERAL}))

        assert getattr(exc, "held") is value

    @pytest.mark.parametrize("payload", [f"x {_LITERAL}", [f"x {_LITERAL}"]], ids=["str", "list"])
    def test_a_property_shadowing_an_instance_attribute_does_not_raise(self, payload: Any) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        class PropertyError(Exception):
            @property
            def info(self) -> Any:
                return self.__dict__["info"]

        exc = PropertyError("failed")
        exc.__dict__["info"] = payload

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in repr(exc.__dict__["info"])

    def test_a_tuple_first_cycle_in_a_tuple_attribute_is_masked(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        lst: list[Any] = [_LITERAL]
        cyc = (lst,)
        lst.append(cyc)
        exc = ValueError("failed")
        setattr(exc, "held", cyc)

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in repr(getattr(exc, "held"))

    def test_a_tuple_first_cycle_in_an_arg_is_masked(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        lst: list[Any] = [_LITERAL]
        cyc = (lst,)
        lst.append(cyc)
        exc = ValueError(cyc)

        redact_exception(exc, frozenset({_LITERAL}))

        assert _LITERAL not in repr(exc.args)

    @pytest.mark.parametrize("form", list(_FORM_BUILDERS))
    def test_a_group_message_holding_any_form_returns_a_masked_copy(self, form: str) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        build = _FORM_BUILDERS[form]
        group = ExceptionGroup(f"bad {build(_FORM_LITERAL)} end", [ValueError("m")])

        result = redact_exception(group, frozenset({_FORM_LITERAL}))

        assert result is not group
        assert type(result) is ExceptionGroup
        masked = cast_group(result)
        for form in (_FORM_LITERAL, build(_FORM_LITERAL)):
            assert form not in masked.message
            assert form not in str(masked)
            assert form not in masked.args[0]
        assert masked.exceptions == group.exceptions

    def test_a_group_with_no_message_form_keeps_identity_and_masks_members_in_place(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        member = ValueError(f"m {_LITERAL}")
        group = ExceptionGroup("clean", [member])

        result = redact_exception(group, frozenset({_LITERAL}))

        assert result is group
        assert group.exceptions[0] is member
        assert _LITERAL not in str(member)

    @pytest.mark.parametrize("flag", [True, False])
    def test_the_copy_keeps_the_exception_state(self, flag: bool) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        member = ValueError("m")
        group = ExceptionGroup(f"bad {_LITERAL}", [member])
        try:
            raise group
        except ExceptionGroup:
            pass
        context = KeyError("ctx")
        cause = LookupError("cause")
        group.__context__ = context
        if flag:
            group.__cause__ = cause
        else:
            group.__suppress_context__ = False
        group.add_note("note")
        setattr(group, "extra", {"k": 1})
        traceback_before = group.__traceback__

        result = redact_exception(group, frozenset({_LITERAL}))

        assert result is not group
        assert result.__traceback__ is traceback_before
        assert result.__context__ is context
        assert result.__cause__ is (cause if flag else None)
        assert result.__suppress_context__ is flag
        assert getattr(result, "__notes__") == ["note"]
        assert getattr(result, "extra") == {"k": 1}
        assert result.__dict__["extra"] == {"k": 1}
        assert isinstance(result, BaseExceptionGroup)
        assert result.exceptions[0] is member

    def test_the_copy_keeps_a_context_with_no_cause_and_a_false_flag(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        group = ExceptionGroup(f"bad {_LITERAL}", [ValueError("m")])
        context = KeyError("ctx")
        group.__context__ = context
        assert group.__suppress_context__ is False

        result = redact_exception(group, frozenset({_LITERAL}))

        assert result is not group
        assert result.__context__ is context
        assert result.__cause__ is None
        assert result.__suppress_context__ is False

    @pytest.mark.parametrize("layout", ["code_first", "code_last"])
    @pytest.mark.parametrize("member_changed", [True, False])
    def test_the_members_arg_keeps_its_position_under_a_custom_layout(self, layout: str, member_changed: bool) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        member = ValueError(f"m {_LITERAL}" if member_changed else "m")
        message = f"bad {_LITERAL}"
        group = ExceptionGroup(message, [member])
        original_members = group.exceptions
        group.args = (7, message, original_members) if layout == "code_first" else (message, original_members, 7)

        result = redact_exception(group, frozenset({_LITERAL}))

        assert result is not group
        code_at, message_at, members_at = (0, 1, 2) if layout == "code_first" else (2, 0, 1)
        assert result.args[code_at] == 7
        assert _LITERAL not in result.args[message_at]
        copied_members = result.args[members_at]
        assert len(result.args) == 3
        assert copied_members is not original_members
        assert tuple(copied_members) == cast_group(result).exceptions
        assert all(_LITERAL not in str(item) for item in copied_members)

    def test_a_custom_new_group_is_rebuilt_without_calling_new_and_pickles_masked(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        group = _CodedGroup(f"bad {_LITERAL}", [ValueError("m")], 42)

        result = redact_exception(group, frozenset({_LITERAL}))

        assert result is not group
        assert type(result) is _CodedGroup
        assert getattr(result, "code") == 42
        assert _LITERAL not in cast_group(result).message
        restored = pickle.loads(pickle.dumps(result))  # nosec B301
        assert type(restored) is _CodedGroup
        assert _LITERAL not in restored.message
        assert restored.code == 42

    def test_a_frozen_dataclass_group_does_not_raise(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        group = _FrozenGroup(f"bad {_LITERAL}", [ValueError("m")])

        result = redact_exception(group, frozenset({_LITERAL}))

        assert type(result) is _FrozenGroup
        assert _LITERAL not in cast_group(result).message

    def test_a_nested_secret_group_rebuilds_a_clean_parent(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        inner = ExceptionGroup(f"in {_LITERAL}", [ValueError("m")])
        parent = ExceptionGroup("clean", [inner])

        result = cast_group(redact_exception(parent, frozenset({_LITERAL})))

        assert result is not parent
        assert result.exceptions[0] is not inner
        assert _LITERAL not in cast_group(result.exceptions[0]).message
        assert tuple(result.args[1]) == result.exceptions
        assert _LITERAL not in "".join(_group_texts(result))

    def test_a_group_shared_by_two_parents_maps_to_one_copy(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        shared = ExceptionGroup(f"s {_LITERAL}", [ValueError("m")])
        root = ExceptionGroup("root", [ExceptionGroup("a", [shared]), ExceptionGroup("b", [shared])])

        result = cast_group(redact_exception(root, frozenset({_LITERAL})))

        first, second = (cast_group(item) for item in result.exceptions)
        assert first.exceptions[0] is second.exceptions[0]
        assert first.exceptions[0] is not shared
        assert _LITERAL not in "".join(_group_texts(result))

    @pytest.mark.parametrize("link", ["__cause__", "__context__"])
    def test_a_group_in_a_chain_slot_is_swapped_keeping_the_plain_exception(self, link: str) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        group = ExceptionGroup(f"bad {_LITERAL}", [ValueError("m")])
        plain = RuntimeError("plain")
        setattr(plain, link, group)
        flag = plain.__suppress_context__

        result = redact_exception(plain, frozenset({_LITERAL}))

        assert result is plain
        swapped = getattr(plain, link)
        assert swapped is not group
        assert _LITERAL not in swapped.message
        assert plain.__suppress_context__ is flag

    def test_a_chain_cycle_through_a_group_terminates(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        group = ExceptionGroup(f"bad {_LITERAL}", [ValueError("m")])
        plain = RuntimeError("plain")
        plain.__cause__ = group
        group.__context__ = plain

        result = redact_exception(plain, frozenset({_LITERAL}))

        assert result is plain
        assert _LITERAL not in "".join(_group_texts(plain))

    def test_a_group_in_refusals_and_in_a_plain_exceptions_attribute_is_swapped(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception
        from mloda.core.prepare.identify_feature_group import FeatureResolutionError

        group = ExceptionGroup(f"bad {_LITERAL}", [ValueError("m")])
        error = FeatureResolutionError("msg", "f", None, refusals=[group])  # type: ignore[arg-type]
        redact_exception(error, frozenset({_LITERAL}))
        assert error.refusals[0] is not group
        assert _LITERAL not in cast_group(error.refusals[0]).message

        other = ExceptionGroup(f"bad {_LITERAL}", [ValueError("m")])
        holder = RuntimeError("h")
        setattr(holder, "exceptions", [other])
        result = redact_exception(holder, frozenset({_LITERAL}))
        assert result is holder
        swapped = getattr(holder, "exceptions")[0]
        assert swapped is not other
        assert _LITERAL not in swapped.message

    def test_nesting_deeper_than_the_depth_bound_leaves_no_secret(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import _MAX_MASK_DEPTH, redact_exception

        node: BaseException = ExceptionGroup(f"deep {_LITERAL}", [ValueError(f"m {_LITERAL}")])
        for _ in range(_MAX_MASK_DEPTH + 8):
            node = BaseExceptionGroup("clean", [node])

        result = redact_exception(node, frozenset({_LITERAL}))

        assert all(_LITERAL not in text for text in _group_texts(result))

    def test_except_star_still_splits_the_rebuilt_group(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import redact_exception

        member = ValueError("m")
        result = redact_exception(ExceptionGroup(f"bad {_LITERAL}", [member, KeyError("k")]), frozenset({_LITERAL}))

        caught: list[Any] = []
        try:
            raise result
        except* ValueError as split:
            caught.append(split)
        except* KeyError:
            pass

        assert caught[0].exceptions == (member,)
        assert _LITERAL not in caught[0].message

    def test_reraise_redacted_does_not_raise_when_nothing_was_rebuilt(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import (  # type: ignore[attr-defined,unused-ignore]
            reraise_redacted,
        )

        group = ExceptionGroup("clean", [ValueError(f"m {_LITERAL}")])
        plain = ValueError(_LITERAL)

        reraise_redacted(group, frozenset({_LITERAL}))
        reraise_redacted(plain, frozenset({_LITERAL}))

        assert all(_LITERAL not in str(member) for member in group.exceptions)
        assert all(_LITERAL not in str(arg) for arg in plain.args)

    @pytest.mark.parametrize("outer", [False, True])
    @pytest.mark.parametrize("frozen", [False, True])
    def test_redact_escaping_exception_raises_the_masked_copy(self, outer: bool, frozen: bool) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import (
            redact_escaping_exception,
            registered_secrets_scope,
        )

        def make() -> BaseException:
            args = (f"bad {_LITERAL}", [ValueError("m")])
            return _FrozenGroup(*args) if frozen else ExceptionGroup(*args)

        original_context = KeyError("ctx")
        raised: list[BaseException] = []
        caught: BaseException | None = None

        def body() -> None:
            try:
                with registered_secrets_scope(frozenset({_LITERAL})):
                    with redact_escaping_exception():
                        group = make()
                        raised.append(group)
                        if outer:
                            object.__setattr__(group, "__context__", original_context)
                        raise group
            except BaseException as error:
                nonlocal caught
                caught = error

        if outer:
            try:
                raise original_context
            except KeyError:
                body()
        else:
            body()

        assert caught is not None
        assert caught is not raised[0]
        assert type(caught) is type(raised[0])
        assert _LITERAL not in cast_group(caught).message
        assert caught.__context__ is raised[0].__context__
        assert caught.__context__ is not raised[0]
        assert _LITERAL not in "".join(traceback.format_exception(caught))
