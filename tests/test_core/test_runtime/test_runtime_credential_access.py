"""Feature groups read explicit credential handles only while a runtime hook runs."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import threading
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.credential import RegisteredCredential
from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import (
    Credential,
    DataAccessCollection,
    Feature,
    FeatureName,
    Options,
    ParallelizationMode,
    PluginCollector,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (  # noqa: F401
    PythonDictFramework,
)


_HANDLE = "runtime-service"
_HOOK_VALUE = "opaque-runtime-material-2351"
_HOOK_BYTES = b"opaque-runtime-binary-2351"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _collection(value: str = _HOOK_VALUE) -> DataAccessCollection:
    return DataAccessCollection(
        credentials={
            _HANDLE: Credential(material=value, binary=_HOOK_BYTES, lifetime=3600),
            "another-service": Credential(material="unselected-material"),
        }
    )


def _request(group: type[FeatureGroup], collection: DataAccessCollection | None, **options: Any) -> dict[str, Any]:
    return {
        "features": [Feature(group.get_class_name(), options=options)],
        "data_access_collection": collection,
        "compute_frameworks": ["PythonDictFramework"],
        "plugin_collector": PluginCollector.enabled_feature_groups({group, RuntimeCredentialSeed}),
    }


def _assert_hook_credential(group: type[FeatureGroup]) -> str:
    credential = group.read_credential(_HANDLE)
    assert isinstance(credential, RegisteredCredential)
    assert credential["material"] == _HOOK_VALUE
    assert credential["binary"] == _HOOK_BYTES
    assert credential["lifetime"] == 3600
    for rendered in (str(credential), repr(credential)):
        assert _HOOK_VALUE not in rendered
        assert _HOOK_BYTES.decode() not in rendered
    return _digest(credential["material"])


class RuntimeCredentialSeed(DataCreatorFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["seed"]}


class RuntimeCredentialHooks(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        return {Feature(RuntimeCredentialSeed.get_class_name())}

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> None:
        data[RuntimeCredentialSeed.get_class_name()][0] = _assert_hook_credential(cls)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        digest = _assert_hook_credential(cls)
        assert data[RuntimeCredentialSeed.get_class_name()] == [digest]
        evidence = f"{digest}|{os.getpid()}|{multiprocessing.get_start_method(allow_none=True)}|{threading.get_ident()}"
        return {cls.get_class_name(): [evidence]}

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        assert data[cls.get_class_name()][0].startswith(_assert_hook_credential(cls))
        data[cls.get_class_name()][0] += "|validated"


class RuntimeCredentialLookup(DataCreatorFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        credential = cls.read_credential(features.get_options_key("credential_handle") or _HANDLE)
        return {cls.get_class_name(): [_digest(credential["material"])]}


class RuntimeCredentialDuringPlanning(RuntimeCredentialLookup):
    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        self.read_credential(_HANDLE)


@pytest.mark.parametrize(
    "mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING]
)
def test_all_runtime_hooks_read_explicit_credentials_without_extenders(
    mode: ParallelizationMode, flight_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLODA_MP_START_METHOD", "spawn")
    result = mloda.run_all(
        **_request(RuntimeCredentialHooks, _collection()),
        parallelization_modes={mode},
        flight_server=flight_server,
    )

    digest, pid, start_method, thread_id, validated = result[0][RuntimeCredentialHooks.get_class_name()][0].split("|")
    assert digest == _digest(_HOOK_VALUE)
    assert validated == "validated"
    if mode is ParallelizationMode.MULTIPROCESSING:
        assert int(pid) != os.getpid()
        assert start_method == "spawn"
    elif mode is ParallelizationMode.THREADING:
        assert int(pid) == os.getpid()
        assert int(thread_id) != threading.get_ident()
    else:
        assert int(pid) == os.getpid()
        assert int(thread_id) == threading.get_ident()


def test_read_credential_is_unavailable_outside_runtime() -> None:
    with pytest.raises(RuntimeError, match="(?i)(runtime|hook|active)"):
        FeatureGroup.read_credential(_HANDLE)


def test_read_credential_is_unavailable_during_preparation() -> None:
    with pytest.raises(RuntimeError, match="(?i)(runtime|hook|active)"):
        mloda.prepare(**_request(RuntimeCredentialDuringPlanning, _collection()))


@pytest.mark.parametrize("handle", ["missing-service", "file-service", "folder-service", "connection-service"])
def test_read_credential_preserves_safe_resolver_errors(handle: str) -> None:
    collection = DataAccessCollection(
        credentials={_HANDLE: Credential(material=_HOOK_VALUE)},
        files={"file-service": "unused-file.csv"},
        folders={"folder-service": "unused-folder"},
        connections={"connection-service": object()},
    )
    with pytest.raises(ValueError) as expected:
        collection.resolve("credentials", hint=handle)

    with pytest.raises(ValueError) as actual:
        mloda.run_all(**_request(RuntimeCredentialLookup, collection, credential_handle=handle))

    assert str(actual.value) == str(expected.value)
    assert _HOOK_VALUE not in str(actual.value)


def test_read_credential_without_a_collection_reports_the_missing_handle() -> None:
    with pytest.raises(ValueError) as exc_info:
        mloda.run_all(**_request(RuntimeCredentialLookup, None))

    assert _HANDLE in str(exc_info.value)
    assert "credential" in str(exc_info.value).lower()
