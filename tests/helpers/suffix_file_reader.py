"""Test-local BaseInputData file reader that drives the suffix, column and pin matching machinery.

A subclass owns unique suffixes, so it never claims data in other tests. It stays non-final until a
subclass overrides ``load_data``. A reader matches only itself, so a root FeatureGroup returns the concrete reader.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    CHAIN_SEPARATOR,
    COLUMN_SEPARATOR,
)
from mloda.core.abstract_plugins.components.utils import is_match_abort
from mloda.provider import INPUT_DATA_STAGE, BaseInputData, record_match_rejection
from mloda.user import DataAccessCollection, Options


class SuffixFileReader(BaseInputData):
    """Family base: suffix ownership, column validation, separator decline and pin handling."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        raise NotImplementedError

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        raise NotImplementedError

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if isinstance(data_access, DataAccessCollection):
            file_match = data_access.resolve(
                "file", predicate=lambda p: cls._file_matches(p, feature_names), hint=options.get("data_access_handle")
            )
            if file_match is not None:
                return file_match
            return cls._match_paths(list(data_access.folders.values()), feature_names)
        if isinstance(data_access, str):
            return cls._match_paths([data_access], feature_names)
        if isinstance(data_access, Path):
            return cls._match_paths([str(data_access)], feature_names)
        return None

    @classmethod
    def _match_paths(cls, data_accesses: list[str], feature_names: list[str]) -> Any:
        for data_access in data_accesses:
            if data_access.endswith(cls.suffix()):
                if cls._file_matches(data_access, feature_names):
                    return data_access
                continue
            if os.path.isdir(data_access):
                for file in os.listdir(data_access):
                    file_name = os.path.join(data_access, file)
                    if cls._file_matches(file_name, feature_names):
                        return file_name
        return None

    @classmethod
    def _file_matches(cls, path: str, feature_names: list[str]) -> bool:
        if not path.endswith(cls.suffix()):
            return False
        try:
            if cls.validate_columns(path, feature_names) is False:
                return False
            return not cls._declines_unvalidated_separator_name(path, feature_names)
        except (OSError, ValueError) as exc:
            if is_match_abort(exc):
                raise
            record_match_rejection(
                cls.get_class_name(),
                f"{cls.get_class_name()} matched the suffix of {path} but could not read its columns: {exc}",
                stage=INPUT_DATA_STAGE,
            )
            return False

    @classmethod
    def _declines_unvalidated_separator_name(cls, file_name: str, feature_names: list[str]) -> bool:
        feature = next((name for name in feature_names if CHAIN_SEPARATOR in name or COLUMN_SEPARATOR in name), None)
        if feature is None or cls._column_names_or_none(file_name) is not None:
            return False
        record_match_rejection(
            cls.get_class_name(),
            f"{cls.get_class_name()} matched the suffix of {file_name} but cannot enumerate its columns via "
            f"get_column_names, so it cannot confirm the chain/column-separated name '{feature}'",
            stage=INPUT_DATA_STAGE,
        )
        return True

    @classmethod
    def _column_names_or_none(cls, file_name: str) -> list[str] | None:
        try:
            return cls.get_column_names(file_name)
        except (NotImplementedError, ImportError) as exc:
            if is_match_abort(exc):
                raise
            return None

    @classmethod
    def validate_columns(cls, file_name: str, feature_names: list[str]) -> bool:
        columns = cls._column_names_or_none(file_name)
        if columns is None:
            return True
        missing = [feature for feature in feature_names if feature not in columns]
        if missing:
            record_match_rejection(
                cls.get_class_name(),
                f"{cls.get_class_name()} matched the suffix of {file_name} but it lacks the column(s): "
                f"{', '.join(missing)}",
                stage=INPUT_DATA_STAGE,
            )
            return False
        return True
