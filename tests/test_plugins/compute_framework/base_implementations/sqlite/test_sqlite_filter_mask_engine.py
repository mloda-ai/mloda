from typing import Any

import pytest

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from tests.test_plugins.compute_framework.base_implementations.filter_mask_engine_test_mixin import (
    FilterMaskEngineTestMixin,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_filter_mask_engine import (
    SqliteFilterMaskEngine,
)


class TestSqliteFilterMaskEngine(FilterMaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseFilterMaskEngine]:
        return SqliteFilterMaskEngine

    @pytest.fixture
    def sample_data(self, connection: Any) -> Any:
        return SqliteRelation.from_dict(
            connection,
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            },
        )

    def mask_to_list(self, mask: Any) -> list[bool]:
        return list(mask)
