from typing import Any, Optional
import pytest

from mloda.provider import BaseMergeEngine
from mloda.user import Index, JoinType
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_merge_engine import PyArrowMergeEngine
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link
from tests.test_plugins.compute_framework.test_tooling.multi_index.multi_index_test_base import (
    MultiIndexMergeEngineTestBase,
)

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowMergeEngineHelperColumnCollision:
    """Regression: the different-index-name path must not collide with a user column."""

    def test_merge_inner_different_index_names_with_existing_mloda_right_index(self) -> None:
        """Merging with different left/right index names must not collide with a user column.

        The different-index-name path copies the right index into an internal helper column.
        When a user column is already named ``mloda_right_index``, the merge must still succeed,
        preserve that user column's values, and not leak the helper into the output schema.
        """
        left = pa.Table.from_pydict({"left_id": [1, 2, 3], "lval": ["a", "b", "c"]})
        right = pa.Table.from_pydict(
            {
                "right_id": [2, 3, 4],
                "mloda_right_index": [99, 88, 77],  # user column named like the internal helper
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
        # The synthetic join-key helper must NOT leak into the output schema.
        assert set(result.column_names) == {"left_id", "lval", "right_id", "mloda_right_index", "rval"}

    @pytest.mark.parametrize("jointype", [JoinType.INNER, JoinType.LEFT, JoinType.RIGHT, JoinType.OUTER])
    def test_merge_differing_keys_with_shared_payload_column(self, jointype: JoinType) -> None:
        """Differing key names plus a SHARED non-key column name must not crash the helper drop.

        Arrow's join result then holds two same-named payload columns, so dropping helpers by
        name would raise ``KeyError: Field "value" exists 2 times``. Dropping by index avoids it,
        and both original key columns must survive.
        """
        left = pa.Table.from_pydict({"lk": [1, 2, 3], "value": ["a", "b", "c"]})
        right = pa.Table.from_pydict({"rk": [1, 2, 4], "value": ["x", "y", "z"]})

        result = PyArrowMergeEngine().merge(left, right, make_merge_link(jointype, Index(("lk",)), Index(("rk",))))

        assert "lk" in result.column_names
        assert "rk" in result.column_names


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
