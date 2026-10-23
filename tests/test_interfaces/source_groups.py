"""The stock file source groups the interface tests enable in their plugin collectors."""

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from mloda_plugins.feature_group.input_data.file_formats.feather_fg import FeatherFG
from mloda_plugins.feature_group.input_data.file_formats.json_fg import JsonFG
from mloda_plugins.feature_group.input_data.file_formats.orc_fg import OrcFG
from mloda_plugins.feature_group.input_data.file_formats.parquet_fg import ParquetFG

FILE_SOURCE_GROUPS: frozenset[type[FeatureGroup]] = frozenset({CsvFG, FeatherFG, JsonFG, OrcFG, ParquetFG})
