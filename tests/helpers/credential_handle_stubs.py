"""FeatureGroups and keys shared by the credential handle tests across several test files."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

from mloda.provider import (
    BaseArtifact,
    ComputeFramework,
    DataCreatorFeatureGroup,
    FeatureSet,
    credential_handle_spec,
    property_spec,
)
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.credential_grant import CredentialLookup

HANDLE_KEY = "chs_vault_token"


class ChsSource(DataCreatorFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({name: [1, 2, 3] for name in features.get_all_names()})


class ChsVaultFG(ChsSource):
    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_vault_a", "chs_vault_b"}


class ChsRawSecretFG(ChsSource):
    PROPERTY_MAPPING = {HANDLE_KEY: property_spec("Raw secret, not a handle.", default=None)}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_raw_a"}


def _granted(handle: str) -> Any:
    from mloda.core.abstract_plugins.components.credential_grant import _read_granted

    return _read_granted(handle)


def _secret_field(value: Mapping[str, Any]) -> str:
    return str(value["hmac"] if "hmac" in value else value["token"])


def _secret_length_via_helper(handle: str) -> int:
    return len(_secret_field(_granted(handle)))


class ChsReadingFG(ChsSource):
    """Reads its declared handle through the private reader, in calculate_feature and in a helper."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_read_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        handle = features.get_options_key(HANDLE_KEY)
        host = _granted(handle)["host"]
        row = f"{host}|{_secret_length_via_helper(handle)}"
        return pd.DataFrame({name: [row] for name in features.get_all_names()})


class ChsFailingReadFG(ChsSource):
    """Reads its declared handle, then fails with an error that echoes the host and the secret."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_failing_read_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        value = _granted(features.get_options_key(HANDLE_KEY))
        raise ValueError(f"read failed host={value['host']} value={dict(value)}")


class ChsReadingArtifact(BaseArtifact):
    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Any | None:
        return {"saved_host": _granted(features.get_options_key(HANDLE_KEY))["host"]}


class ChsArtifactFG(ChsSource):
    """Its artifact saver reads the declared handle, so the grant must span save_artifact."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_artifact_a"}

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return ChsReadingArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.artifact_to_save:
            features.save_artifact = "chs-artifact"
        return pd.DataFrame({name: [1] for name in features.get_all_names()})


GRANT_ONLY_SECRET = "grant-only-Zk3Fw8Qd2"  # nosec B105


class GrantOnlyKind:
    """Test grant kind: every declared handle maps to a value only this class knows."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self.handles = tuple(values)

    def open(self) -> "CredentialLookup":
        from mloda.core.abstract_plugins.components.credential_grant import CredentialLookup

        return CredentialLookup({h: {"host": "grant-only.example", "token": GRANT_ONLY_SECRET} for h in self.handles})
