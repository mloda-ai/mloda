"""Per-framework implementations of the three column-wise hooks, shared by every family.

This module sits outside ``experimental/`` on purpose: the hook sweep binds every hook-calling
module under that tree to a family base directory, and a framework adapter belongs to no family.
"""

from __future__ import annotations

from typing import Any, cast

from mloda.provider import FeatureGroup
from mloda.user.python_dict import row_count


class ColumnwiseHooks:
    """Framework-neutral source-feature check, branching on the family's presence policy."""

    # True: raise as soon as ANY source name is missing. False: raise only when NONE of them exists.
    STRICT_SOURCE_FEATURES: bool = True

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        """Get the set of available column names from the data."""
        raise NotImplementedError(f"{cls.__name__} must implement _get_available_columns")

    @classmethod
    def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
        """Check that the resolved source features exist, as strictly as the family declares."""
        available = cls._get_available_columns(data)
        missing = [name for name in feature_names if name not in available]
        if cls.STRICT_SOURCE_FEATURES:
            if missing:
                raise ValueError(
                    f"Source features not found in data: {missing}. Available columns: {sorted(available)}"
                )
        elif len(missing) == len(feature_names):
            raise ValueError(
                f"None of the source features {feature_names} found in data. Available columns: {sorted(available)}"
            )


class PandasColumnwiseHooks(ColumnwiseHooks):
    """Column-wise hooks for a pandas DataFrame."""

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        """Get the set of available column names from the DataFrame."""
        return set(data.columns)

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the result to the DataFrame."""
        data[feature_name] = result
        return data


class SklearnPandasColumnwiseHooks(PandasColumnwiseHooks):
    """Writer for a scikit-learn result, spreading a multi-column array over the ~N naming convention."""

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the result to the DataFrame, one column per result dimension."""
        if hasattr(result, "shape") and len(result.shape) == 2:
            if result.shape[1] == 1:
                data[feature_name] = result.flatten()
            else:
                # cls is always a FeatureGroup subclass at call time; this class alone is not one.
                named_columns = cast("type[FeatureGroup]", cls).apply_naming_convention(result, feature_name)
                for col_name, col_data in named_columns.items():
                    data[col_name] = col_data
        elif hasattr(result, "shape") and len(result.shape) == 1:
            data[feature_name] = result
        else:
            data[feature_name] = result

        return data


class PyArrowColumnwiseHooks(ColumnwiseHooks):
    """Column-wise hooks for a PyArrow Table, duck-typed so this module needs no pyarrow import."""

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        """Get the set of available column names from the Table schema."""
        return set(data.schema.names)

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the result to the Table, replacing the column when it already exists."""
        if feature_name in data.schema.names:
            column_index = data.schema.names.index(feature_name)
            data = data.remove_column(column_index)
            return data.append_column(feature_name, result)
        return data.append_column(feature_name, result)


class PythonDictColumnwiseHooks(ColumnwiseHooks):
    """Column-wise hooks for a columnar ``dict[str, list[Any]]``."""

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        """Get the set of available column names from the data."""
        return set(data.keys())

    @classmethod
    def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
        """Reject empty data before the presence policy runs."""
        if not data:
            raise ValueError("Data cannot be empty")
        super()._check_source_features_exist(data, feature_names)

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the row-aligned result to the data."""
        if len(result) != row_count(data):
            raise ValueError(f"Result length {len(result)} does not match data length {row_count(data)}")

        data[feature_name] = list(result)

        return data


class PolarsLazyColumnwiseHooks(ColumnwiseHooks):
    """Column-wise hooks for a Polars LazyFrame."""

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        """Get the set of available column names from the LazyFrame schema."""
        if not hasattr(data, "collect_schema"):
            raise ValueError("Data does not have a collect_schema method, cannot get available columns.")
        return set(data.collect_schema().names())
