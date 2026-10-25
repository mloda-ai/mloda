"""The stock file source groups, and a deliberately failing group, the interface tests enable in their collectors."""

from typing import Any

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.validators.missing_input_validator import MissingInputFeatureError
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


PLAN_RAISE = "mcp5_plan_raise"
PLAN_RAISE_SECRET = "SECRET_plan_raise_9e2b"  # nosec B105


class PlanRaiseMcp5(FeatureGroup):
    """Fails while planning, with a message that holds data."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == PLAN_RAISE

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        raise ValueError(PLAN_RAISE_SECRET)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data


FAILED = "income__mcp5_failed"
FAILED_MANY = "income__mcp5_failed_many"
FAILED_CTRL = "income__mcp5_failed_ctrl"
FAILED_LONG = "income__mcp5_failed_long"
FAILED_ENTRIES = "income__mcp5_failed_entries"
FAILED_COLUMN_CTRL = "bad\ncol"
FAILED_COLUMN_LONG = "c" * 250
FAILED_MANY_ROWS = 1500
FAILED_ENTRY_COUNT = 120


class RunnerFailedRowsMcp5(FeatureGroup):
    """Derives income and reports rows that got no value."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) in {FAILED, FAILED_MANY, FAILED_CTRL, FAILED_LONG, FAILED_ENTRIES}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("income")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        from mloda.provider import report_failed_rows

        columns: dict[str, list[Any]] = {}
        for feature_name in features.get_all_names():
            name = str(feature_name)
            columns[name] = [int(value) for value in data["income"]]
            if name == FAILED:
                report_failed_rows(name, [1, 3], "timeout")
            elif name == FAILED_MANY:
                report_failed_rows(name, list(range(FAILED_MANY_ROWS)), "bulk")
            elif name == FAILED_LONG:
                report_failed_rows(FAILED_COLUMN_LONG, [0], "long")
            elif name == FAILED_ENTRIES:
                for index in range(FAILED_ENTRY_COUNT):
                    report_failed_rows(f"col{index}", [0], "many")
            else:
                report_failed_rows(FAILED_COLUMN_CTRL, [0], "ctrl")
        return columns


MISSING = "income__mcp5_missing"
MISSING_INPUT = "mcp5_absent_input"


class RunnerMissingInputMcp5(FeatureGroup):
    """Fails while computing because a declared input is absent."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == MISSING

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("income")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise MissingInputFeatureError("RunnerMissingInputMcp5", MISSING, (MISSING_INPUT,), ("income",))
