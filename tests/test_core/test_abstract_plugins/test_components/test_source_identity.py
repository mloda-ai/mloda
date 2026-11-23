"""Source identities use the public file-source recipe independently of registered credentials."""

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope
from mloda.core.abstract_plugins.components.input_data.file_source_fg import _FileSourceFG
from mloda.user import DataAccessCollection


def test_source_identity_is_public() -> None:
    assert "source_identity" in provider.__all__
    assert callable(provider.source_identity)


@pytest.mark.parametrize(
    "uri, expected",
    [
        pytest.param(
            "s3://alice:uri-secret@bucket/data.csv?version=4#part",
            "s3://bucket/data.csv",
            id="userinfo-query-fragment",
        ),
        pytest.param(
            "https://files.example.com:8443/report@2026.csv",
            "https://files.example.com:8443/report@2026.csv",
            id="port-and-path-at-sign",
        ),
        pytest.param(
            "abfs://container@account.dfs.core.windows.net/data.csv",
            "abfs://container@account.dfs.core.windows.net/data.csv",
            id="azure-container",
        ),
        pytest.param(
            "https://files.example.com/password=path-secret/data.csv",
            "https://files.example.com/password=***/data.csv",
            id="credential-shaped-path-segment",
        ),
        pytest.param(
            "https://files.example.com/folder%3Bsession=path-secret/data.csv",
            "https://files.example.com/folder;***/data.csv",
            id="encoded-path-parameter",
        ),
        pytest.param(
            "simplecache::s3://alice:uri-secret@bucket/folder;session=path-secret/data.csv",
            "simplecache::s3://bucket/folder;***/data.csv",
            id="fsspec-chain-and-path-parameter",
        ),
        pytest.param(
            "s3://AKIA:12/path-secret@bucket/data.csv",
            "s3://AKIA:12/***@bucket/data.csv",
            id="object-store-password-in-path",
        ),
    ],
)
def test_source_identity_matches_file_source_identity(uri: str, expected: str) -> None:
    match = _FileSourceFG._source(uri)
    assert match.source == expected
    assert match.access == uri
    assert provider.source_identity(uri) == match.source


def test_source_identity_keeps_a_relative_path_relative() -> None:
    assert provider.source_identity("folder/data.csv") == "folder/data.csv"


@pytest.mark.parametrize(
    "key, literal, redacted",
    [
        pytest.param("locator", "s3://bucket/dataset-2026/data.csv", "***", id="plugin-locator"),
        pytest.param("password", "dataset-2026", "s3://bucket/***/data.csv", id="secret-path-literal"),
    ],
)
def test_source_identity_ignores_collection_credential_literals(key: str, literal: str, redacted: str) -> None:
    uri = "s3://bucket/dataset-2026/data.csv"
    collection = DataAccessCollection(credentials={"source": {key: literal}})
    assert literal in collection.secret_literals()
    assert provider.scrub_credentials(uri) == uri
    assert provider.source_identity(uri) == uri

    with registered_secrets_scope(collection.secret_literals(), collection.identity_literals()):
        assert provider.scrub_credentials(uri) == redacted
        assert provider.source_identity(uri) == uri
        assert _FileSourceFG._source(uri).source == uri

    assert provider.source_identity(uri) == uri
