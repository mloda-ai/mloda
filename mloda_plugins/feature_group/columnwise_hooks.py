"""Per-framework implementations of the column-wise hooks, shared by every family.

Outside ``experimental/`` on purpose: the hook sweep binds every hook-calling module under that
tree to a family base directory, and a framework adapter belongs to no family.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from mloda.provider import COLUMN_DISCOVERY_HOOKS, FeatureGroup
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import row_count


class ColumnwiseHooks:
    """Framework-neutral source-feature check, branching on the family's presence policy."""

    # True: raise as soon as ANY source name is missing. False: raise only when NONE of them exists.
    STRICT_SOURCE_FEATURES: bool = True

    # The check below routes through the discovery hook, so every class mixing this in owes all three.
    REQUIRED_COLUMNWISE_HOOKS = COLUMN_DISCOVERY_HOOKS

    if TYPE_CHECKING:
        # Declaration only: a runtime body here would shadow the raising default on
        # FeatureChainParserMixin, which missing_columnwise_hooks compares against by identity.
        @classmethod
        def _get_available_columns(cls, data: Any) -> set[str]: ...

    @classmethod
    def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
        """Check that the resolved source features exist, as strictly as the family declares."""
        available = cls._get_available_columns(data)
        missing = [name for name in feature_names if name not in available]
        # Sorted by str so the message is deterministic for every framework and survives mixed-type
        # column labels, at the cost of a pandas frame's natural column order.
        if cls.STRICT_SOURCE_FEATURES:
            if missing:
                raise ValueError(
                    f"Source features not found in data: {missing}. Available columns: {sorted(available, key=str)}"
                )
        elif len(missing) == len(feature_names):
            raise ValueError(
                f"None of the source features {feature_names} found in data."
                f" Available columns: {sorted(available, key=str)}"
            )


class PandasColumnwiseHooks(ColumnwiseHooks):
    """Column-wise hooks for a pandas DataFrame."""

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        return set(data.columns)

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        data[feature_name] = result
        return data


class SklearnPandasColumnwiseHooks(PandasColumnwiseHooks):
    """Writer for a scikit-learn result, spreading a multi-column array over the ~N naming convention."""

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        if hasattr(result, "shape") and len(result.shape) == 2:
            if result.shape[1] == 1:
                data[feature_name] = result.flatten()
            else:
                # cls is always a FeatureGroup subclass at call time; this class alone is not one.
                named_columns = cast("type[FeatureGroup]", cls).apply_naming_convention(result, feature_name)
                for col_name, col_data in named_columns.items():
                    data[col_name] = col_data
        else:
            data[feature_name] = result

        return data


class PyArrowColumnwiseHooks(ColumnwiseHooks):
    """Column-wise hooks for a PyArrow Table, duck-typed so this module needs no pyarrow import."""

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        return set(data.schema.names)

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        if feature_name in data.schema.names:
            data = data.remove_column(data.schema.names.index(feature_name))
        return data.append_column(feature_name, result)


class PythonDictColumnwiseHooks(ColumnwiseHooks):
    """Column-wise hooks for a columnar ``dict[str, list[Any]]``."""

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        return set(data.keys())

    @classmethod
    def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
        if not data:
            raise ValueError("Data cannot be empty")
        super()._check_source_features_exist(data, feature_names)

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        if len(result) != row_count(data):
            raise ValueError(f"Result length {len(result)} does not match data length {row_count(data)}")

        data[feature_name] = list(result)

        return data


class PolarsLazyColumnwiseHooks(ColumnwiseHooks):
    """Column-wise hooks for a Polars LazyFrame."""

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        if not hasattr(data, "collect_schema"):
            raise ValueError("Data does not have a collect_schema method, cannot get available columns.")
        return set(data.collect_schema().names())
