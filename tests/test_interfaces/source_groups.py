"""The stock file source groups the interface tests enable in their plugin collectors."""

from typing import Any

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import FeatureSet
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from mloda_plugins.feature_group.input_data.file_formats.feather_fg import FeatherFG
from mloda_plugins.feature_group.input_data.file_formats.json_fg import JsonFG
from mloda_plugins.feature_group.input_data.file_formats.orc_fg import OrcFG
from mloda_plugins.feature_group.input_data.file_formats.parquet_fg import ParquetFG

FILE_SOURCE_GROUPS: frozenset[type[FeatureGroup]] = frozenset({CsvFG, FeatherFG, JsonFG, OrcFG, ParquetFG})

RAISE = "income__mcp5_raise"
RAISE_SECRET = "SECRET_runner_boom_5c1d"  # nosec B105


class RunnerRaiseMcp5(FeatureGroup):
    """Fails while computing, with a message that holds data."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == RAISE

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("income")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise ValueError(RAISE_SECRET)
