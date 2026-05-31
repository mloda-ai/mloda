from typing import Any, Optional
import pytest

from mloda.provider import BaseMergeEngine
from mloda.user import Index
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_merge_engine import PyArrowMergeEngine
from tests.test_plugins.compute_framework.test_tooling.multi_index.multi_index_test_base import (
    MultiIndexMergeEngineTestBase,
)

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowMergeEngineHelperColumnCollision:
    """Regression: the different-index-name path must not collide with a user column."""

    def test_merge_inner_different_index_names_with_existing_mloda_right_index(self) -> None:
        """When left/right index columns differ, join_logic copies the right index into a helper
        column. It currently hardcodes ``mloda_right_index`` and raises ValueError if that name is
        already a column in right_data. The fix picks a collision-free helper name instead, so the
        merge must succeed AND preserve the user's ``mloda_right_index`` column values.
        """
        left = pa.Table.from_pydict({"left_id": [1, 2, 3], "lval": ["a", "b", "c"]})
        right = pa.Table.from_pydict(
            {
                "right_id": [2, 3, 4],
                "mloda_right_index": [99, 88, 77],  # user column that collides with the legacy helper name
                "rval": ["x", "y", "z"],
            }
        )

        engine = PyArrowMergeEngine()
        result = engine.merge_inner(left, right, Index(("left_id",)), Index(("right_id",)))

        rows = result.to_pylist()
        # Inner join on left_id == right_id matches right_id in {2, 3}
        by_key = {row["left_id"]: row for row in rows}
        assert set(by_key.keys()) == {2, 3}
        # The user's mloda_right_index column must survive with its original values per matched row
        assert by_key[2]["mloda_right_index"] == 99
        assert by_key[3]["mloda_right_index"] == 88


class TestPyArrowMergeEngineMultiIndex(MultiIndexMergeEngineTestBase):
    """Test PyArrowMergeEngine using shared multi-index test scenarios."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        """Return the PyArrowMergeEngine class."""
        return PyArrowMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        """Return pyarrow Table type."""
        if pa is None:
            raise ImportError("PyArrow is not installed")
        # mypy can't infer pa.Table type correctly
        table_type: type[Any] = pa.Table
        return table_type

    def get_connection(self) -> Optional[Any]:
        """PyArrow does not require a connection object."""
        return None

    @pytest.mark.skip(reason="PyArrow does not support UNION operations - see GitHub issue #30950")
    def test_merge_union_with_multi_index(self) -> None:
        """Skip UNION test for PyArrow as it's not supported."""
        pass
