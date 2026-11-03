"""FeatureGroups and keys shared by the credential handle tests across several test files."""

from typing import Any

import pandas as pd

from mloda.provider import (
    ComputeFramework,
    DataCreatorFeatureGroup,
    FeatureGroup,
    FeatureSet,
    credential_handle_spec,
    property_spec,
)
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

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


class ChsTwoHandleFG(ChsSource):
    PROPERTY_MAPPING = {
        "chs_two_z_handle": credential_handle_spec("Second handle."),
        "chs_two_a_handle": credential_handle_spec("First handle."),
    }

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_two_handle_col"}


class ChsConsumerFG(FeatureGroup):
    """Declares a handle and consumes the ChsRawSecretFG column, so every hook fires on its step."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("chs_raw_a")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_consumer"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["chs_consumer"] = data["chs_raw_a"]
        return data
