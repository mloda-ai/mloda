"""Test-only toy source groups over in-memory dict-of-columns sources (no optional dependency).

A source is a credential value ``{TOY_KEY: {column: values}}`` keyed by handle, or the dict-of-columns under
the group's class-name option key. Names carry a ``toyfmt`` marker so the groups stay inert for other tests.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, ClassVar

import pyarrow as pa

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import ClaimRoute, SourceFeatureGroup, NamePolicy, SourceMatch
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

TOY_KEY = "toyfmt_columns"
TOY_HANDLE_OPTION = "data_access_handle"


def toy_dac(**sources: dict[str, list[Any]]) -> DataAccessCollection:
    """One credential handle per keyword, each holding a dict-of-columns toy source."""
    return DataAccessCollection(credentials={handle: {TOY_KEY: cols} for handle, cols in sources.items()})


TOY_TOKEN = "tkn_Zq81Lm5Vx0Rb7Ye3Hw29"  # nosec B105


def toy_token_dac(**sources: dict[str, list[Any]]) -> DataAccessCollection:
    """Like toy_dac, with a registered token string next to each toy source."""
    return DataAccessCollection(
        credentials={handle: {TOY_KEY: cols, "token": TOY_TOKEN} for handle, cols in sources.items()}
    )


def toy_token_request(group: type[ToySourceFG], column: str) -> dict[str, Any]:
    """Keyword arguments that run or plan `column` through only `group` over a token-bearing toy source."""
    return {
        "compute_frameworks": ["PythonDictFramework"],
        "plugin_collector": PluginCollector.enabled_feature_groups({group}),
        "data_access_collection": toy_token_dac(h1={column: [1]}),
    }


def foreign_dac() -> DataAccessCollection:
    return DataAccessCollection(credentials={"toyfmt_foreign": {"unrelated_key": {"toyfmt_col": [1]}}})


def column_values(result: list[Any], column: str) -> list[Any]:
    """Values of column from a run_all result in either columnar or row-wise form."""
    first = result[0]
    if isinstance(first, dict):
        return list(first[column])
    if isinstance(first, list):
        return [row[column] for row in first]
    return list(first[column].to_pylist() if hasattr(first[column], "to_pylist") else first[column])


class ToySourceBase(SourceFeatureGroup):
    """Implements the hooks over in-memory sources; declares no routes, so it never claims."""

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        scoped = options.get(cls.get_class_name())
        if scoped is not None:
            return [SourceMatch(source="scoped:toy", access=scoped)]
        if data_access_collection is None:
            return []
        wanted = options.get(TOY_HANDLE_OPTION)
        found: list[SourceMatch] = []
        for handle, value in sorted(data_access_collection.credentials.items()):
            if wanted is not None and handle != wanted:
                continue
            if isinstance(value, dict) and TOY_KEY in value:
                found.append(SourceMatch(source=f"{handle}:toy", access=value[TOY_KEY]))
        return found

    @classmethod
    def columns(cls, match: SourceMatch) -> Collection[str] | None:
        return list(match.access)

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        return pa.table(dict(match.access))


class ToySourceFG(ToySourceBase):
    """Checked columns, searched on its own."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.CHECKED, True),)


class ToySourceSubFG(ToySourceFG):
    """Subclass of a source group; takes over its claims."""


class ToyNeutralDictFG(ToySourceFG):
    """A source group whose neutral form is a columnar dict, native to the dict framework."""

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        return {name: list(match.access[name]) for name in features.get_all_names()}

    @classmethod
    def neutral_type(cls, match: SourceMatch) -> type | None:
        return dict


class ToyBoundedNeutralDictFG(ToyNeutralDictFG):
    """ToyNeutralDictFG limited to three frameworks, so a default run ignores test-only frameworks."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PyArrowTable, PythonDictFramework}


class ToyOtherSourceFG(ToySourceBase):
    """A second checked, searched group over the same toy sources."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.CHECKED, True),)


class ToyOpenFG(ToySourceBase):
    """Open names, only when pointed at."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.OPEN, False),)


class ToyRequiredOptionFG(ToySourceBase):
    """Checked names; the route applies only when the toyfmt_required option is present."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (
        ClaimRoute("toy", NamePolicy.CHECKED, False, required_options=("toyfmt_required",)),
    )


class ToyDeclaredFG(ToySourceBase):
    """Declared names: one fixed name and one prefix; the class name shortcut is a declared name too."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.DECLARED, True),)

    @classmethod
    def declared_names(cls) -> frozenset[str]:
        return frozenset({"toyfmt_declared_col"})

    @classmethod
    def declared_name_prefixes(cls) -> tuple[str, ...]:
        return ("toyfmt_pre_", "toyfmt_conn__")
