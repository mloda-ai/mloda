"""Importing this module loads every stock format group."""

from mloda_plugins.feature_group.input_data.db_formats.sqlite_fg import SqliteFG
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from mloda_plugins.feature_group.input_data.file_formats.feather_fg import FeatherFG
from mloda_plugins.feature_group.input_data.file_formats.json_fg import JsonFG
from mloda_plugins.feature_group.input_data.file_formats.orc_fg import OrcFG
from mloda_plugins.feature_group.input_data.file_formats.parquet_fg import ParquetFG

__all__ = ["CsvFG", "FeatherFG", "JsonFG", "OrcFG", "ParquetFG", "SqliteFG"]
