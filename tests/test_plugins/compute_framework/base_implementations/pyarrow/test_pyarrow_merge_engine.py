from typing import Any
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


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowMergeEngineNestedColumns:
    """Joins must preserve non-key columns of types Arrow's join rejects, not raise ArrowInvalid."""

    @pytest.mark.parametrize("jointype", [JoinType.INNER, JoinType.LEFT, JoinType.RIGHT, JoinType.OUTER])
    @pytest.mark.parametrize("keys", [("k", "k"), ("lk", "rk")])
    def test_merge_preserves_list_and_struct_columns(self, jointype: JoinType, keys: tuple[str, str]) -> None:
        left_key, right_key = keys
        list_type = pa.list_(pa.string())
        struct_type = pa.struct([("x", pa.int64())])

        left = pa.Table.from_pydict(
            {
                left_key: [1, 2, 3],
                "ls": pa.array([["a"], [], ["c", "d"]], type=list_type),
            }
        )
        right = pa.Table.from_pydict(
            {
                right_key: [1, 2, 4],
                "rs": pa.array([["x"], ["y", "z"], []], type=list_type),
                "st": pa.array([{"x": 10}, {"x": 20}, {"x": 30}], type=struct_type),
            }
        )

        expected: dict[int, dict[str, Any]] = {
            1: {"ls": ["a"], "rs": ["x"], "st": {"x": 10}},
            2: {"ls": [], "rs": ["y", "z"], "st": {"x": 20}},
            3: {"ls": ["c", "d"], "rs": None, "st": None},
            4: {"ls": None, "rs": [], "st": {"x": 30}},
        }
        expected_keys_by_type = {
            JoinType.INNER: {1, 2},
            JoinType.LEFT: {1, 2, 3},
            JoinType.RIGHT: {1, 2, 4},
            JoinType.OUTER: {1, 2, 3, 4},
        }

        result = PyArrowMergeEngine().merge(
            left, right, make_merge_link(jointype, Index((left_key,)), Index((right_key,)))
        )

        assert result.schema.field("ls").type == list_type
        assert result.schema.field("rs").type == list_type
        assert result.schema.field("st").type == struct_type

        expected_columns = {"ls", "rs", "st"}
        expected_columns |= {"k"} if left_key == right_key else {"lk", "rk"}
        assert set(result.column_names) == expected_columns

        rows = result.to_pylist()
        by_key: dict[Any, dict[str, Any]] = {}
        for row in rows:
            key = row["k"] if left_key == right_key else (row["lk"] if row["lk"] is not None else row["rk"])
            by_key[key] = row

        assert set(by_key.keys()) == expected_keys_by_type[jointype]
        for key, row in by_key.items():
            assert row["ls"] == expected[key]["ls"]
            assert row["rs"] == expected[key]["rs"]
            assert row["st"] == expected[key]["st"]

    @pytest.mark.parametrize(
        "type_name",
        [
            "null",
            "string_view",
            "binary_view",
            "run_end_encoded",
            "fixed_shape_tensor",
            "list_string_view",
            "dictionary_of_list",
        ],
        ids=[
            "null",
            "string_view",
            "binary_view",
            "run_end_encoded",
            "fixed_shape_tensor",
            "list_string_view",
            "dictionary_of_list",
        ],
    )
    def test_merge_preserves_join_rejected_payload_types(self, type_name: str) -> None:
        """Non-key columns of types Arrow's join rejects must survive an outer join."""

        def build_column(n: int, values: list[str]) -> Any:
            if type_name == "null":
                return pa.array([None] * n, type=pa.null())
            if type_name == "string_view":
                return pa.array(values, type=pa.string_view())
            if type_name == "binary_view":
                return pa.array([v.encode() for v in values], type=pa.binary_view())
            if type_name == "run_end_encoded":
                # Multi-row run (run ends [2, 3]) exercises decode/re-encode beyond length-1 runs.
                logical = [values[0], values[0], values[1]]
                return pa.RunEndEncodedArray.from_arrays(pa.array([2, 3], type=pa.int32()), pa.array(logical))
            if type_name == "fixed_shape_tensor":
                return pa.ExtensionArray.from_storage(
                    pa.fixed_shape_tensor(pa.int64(), [2]),
                    pa.array([[i, i + 1] for i in range(n)], pa.list_(pa.int64(), 2)),
                )
            if type_name == "list_string_view":
                rows = [[values[0]], [], [values[1], values[2]]]
                return pa.array(rows, type=pa.list_(pa.string_view()))
            dictionary = pa.array([[values[0]], [values[1]]], type=pa.list_(pa.string()))
            indices = pa.array([0, 1, 0], type=pa.int32())
            return pa.DictionaryArray.from_arrays(indices, dictionary)

        left_col = build_column(3, ["a", "b", "c"])
        right_col = build_column(3, ["x", "y", "z"])

        left = pa.Table.from_pydict({"k": [1, 2, 3], "lp": left_col})
        right = pa.Table.from_pydict({"k": [1, 2, 4], "rp": right_col})

        result = PyArrowMergeEngine().merge(left, right, make_merge_link(JoinType.OUTER, Index(("k",)), Index(("k",))))

        assert result.schema.field("lp").type == left_col.type
        assert result.schema.field("rp").type == right_col.type
        assert set(result.column_names) == {"k", "lp", "rp"}

        left_values = left_col.to_pylist()
        right_values = right_col.to_pylist()
        left_by_key = dict(zip(left["k"].to_pylist(), left_values))
        right_by_key = dict(zip(right["k"].to_pylist(), right_values))

        rows = result.to_pylist()
        by_key = {row["k"]: row for row in rows}
        assert set(by_key.keys()) == {1, 2, 3, 4}
        for key, row in by_key.items():
            assert row["lp"] == left_by_key.get(key)
            assert row["rp"] == right_by_key.get(key)

    def test_merge_widens_run_end_type_when_join_output_outgrows_it(self) -> None:
        """An int16 run-end type that can no longer address the join output must widen, not error."""
        left = pa.Table.from_pydict(
            {
                "k": [1],
                "lp": pa.RunEndEncodedArray.from_arrays(pa.array([1], type=pa.int16()), pa.array(["a"])),
            }
        )
        right = pa.Table.from_pydict({"k": [1] * 40000})

        result = PyArrowMergeEngine().merge(left, right, make_merge_link(JoinType.INNER, Index(("k",)), Index(("k",))))

        assert result.num_rows == 40000
        assert pa.types.is_run_end_encoded(result.schema.field("lp").type)
        assert result.schema.field("lp").type.value_type == pa.string()
        assert result.column("lp").to_pylist() == ["a"] * 40000


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

    def get_connection(self) -> Any | None:
        """PyArrow does not require a connection object."""
        return None

    @pytest.mark.skip(reason="PyArrow does not support UNION operations - see GitHub issue #30950")
    def test_merge_union_with_multi_index(self) -> None:
        """Skip UNION test for PyArrow as it's not supported."""
        pass
