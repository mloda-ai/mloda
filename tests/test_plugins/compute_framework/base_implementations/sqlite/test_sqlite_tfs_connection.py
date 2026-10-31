"""End-to-end TFS connection-propagation test for SQLite."""

import gc
import sqlite3
import weakref
from typing import Any

import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]

from mloda.user import Feature
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from tests.test_plugins.compute_framework.base_implementations.tfs_connection_e2e_mixin import (
    TfsConnectionEndToEndMixin,
    root_binding_groups,
)


class TfsDoubledSqliteFG(FeatureGroup):
    """Doubles raw_val via a SQLite SELECT that REQUIRES the connection."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SqliteFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.select(_raw_sql="*, raw_val * 2 AS tfs_doubled")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tfs_doubled"}


@pytest.mark.skipif(pa is None, reason="PyArrow not installed (needed for the source FG).")
class TestSqliteTfsConnectionE2E(TfsConnectionEndToEndMixin):
    destination_framework_class = SqliteFramework
    destination_fg_class = TfsDoubledSqliteFG

    @staticmethod
    def double_relation(data: Any, src: str, dst: str) -> Any:
        return data.select(_raw_sql=f"*, {src} * 2 AS {dst}")

    @pytest.fixture
    def live_connection(self) -> Any:
        conn = sqlite3.connect(":memory:")
        yield conn
        conn.close()

    def extract_doubled(self, final: Any) -> list[int]:
        assert isinstance(final, pa.Table)
        assert "tfs_doubled" in final.column_names
        return list(final.column("tfs_doubled").to_pylist())


def test_root_binding_groups_result_is_reclaimed_after_the_caller_releases_it() -> None:
    root, sink, _ = root_binding_groups(SqliteFramework, TestSqliteTfsConnectionE2E.double_relation)
    refs = (weakref.ref(root), weakref.ref(sink))
    assert all(ref() is not None for ref in refs)

    del root, sink
    gc.collect()  # the weakref assert below needs the class cycles actually collected

    assert [ref() is None for ref in refs] == [True, True]
