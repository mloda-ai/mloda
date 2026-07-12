"""``get_transformation_chain`` composes arbitrary registered edges.

Contract:

  * ``ComputeFrameworkTransformer.get_transformation_chain`` resolves a chain by
    breadth-first shortest path over ``transformer_map``, composing ANY registered
    edges, not only the hardcoded ``from -> pa.Table -> to`` two-hop.
  * A direct edge always wins over any multi-hop path (single-element chain).
  * Among multi-hop paths the SHORTEST wins; among equally short paths the tie-break
    is first-registered edge order (insertion order of ``transformer_map``), so the
    result is deterministic across calls in one process.
  * ``None`` is returned when no path exists.
  * The real pa.Table hub (pandas <-> polars) resolves as before.

All synthetic frameworks and transformers are built inside factory helpers, and the
synthetic transformers raise ``ImportError`` from their import hooks so ``add()`` skips
them if global subclass discovery ever sees them. The transformer maps are set
explicitly with BOTH directions per pair, mirroring how ``add()`` registers edges.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
    ComputeFrameworkTransformer,
)

_TransformerMap = dict[tuple[type[Any], type[Any]], type[BaseTransformer]]


def _new_framework(name: str) -> type[Any]:
    """Create a fresh synthetic framework type (a plain class, NOT pa.Table)."""
    return type(name, (), {})


def _make_transformer(
    name: str, fw: type[Any], other_fw: type[Any], forward_tag: str, backward_tag: str
) -> type[BaseTransformer]:
    """Build a synthetic recording transformer for the pair (fw, other_fw).

    The import hooks raise ImportError so ``check_imports`` is False: even if global
    ``BaseTransformer`` subclass discovery sees this class, ``add()`` skips it and the
    class never leaks into another test's registry.
    """

    class _Synthetic(BaseTransformer):
        @classmethod
        def framework(cls) -> Any:
            return fw

        @classmethod
        def other_framework(cls) -> Any:
            return other_fw

        @classmethod
        def import_fw(cls) -> None:
            raise ImportError("synthetic transformer, never auto-registered")

        @classmethod
        def import_other_fw(cls) -> None:
            raise ImportError("synthetic transformer, never auto-registered")

        @classmethod
        def transform_fw_to_other_fw(cls, data: Any) -> Any:
            return [*data, forward_tag]

        @classmethod
        def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
            return [*data, backward_tag]

    _Synthetic.__name__ = name
    _Synthetic.__qualname__ = name
    return _Synthetic


def _map_of(*transformers: type[BaseTransformer]) -> _TransformerMap:
    """Build a transformer map with BOTH directions per pair, in registration order.

    This mirrors ``ComputeFrameworkTransformer.add``: for each transformer the
    (framework, other_framework) edge is inserted first, then the reverse edge.
    Insertion order therefore IS first-registered edge order, which the tie-break
    tests pin.
    """
    transformer_map: _TransformerMap = {}
    for transformer in transformers:
        left = transformer.framework()
        right = transformer.other_framework()
        transformer_map[(left, right)] = transformer
        transformer_map[(right, left)] = transformer
    return transformer_map


def _registry_with(*transformers: type[BaseTransformer]) -> ComputeFrameworkTransformer:
    registry = ComputeFrameworkTransformer()
    registry.transformer_map = _map_of(*transformers)
    return registry


class TestChainCompositionOverRegisteredEdges:
    """BFS composition over arbitrary registered edges."""

    def test_two_hop_composition_over_non_pyarrow_hub(self) -> None:
        """A <-> B and B <-> C registered, B is NOT pa.Table: (A, C) resolves to the
        2-hop chain [T(A,B), T(B,C)].
        """
        fw_a = _new_framework("_FwA")
        fw_b = _new_framework("_FwB")
        fw_c = _new_framework("_FwC")
        trans_ab = _make_transformer("_TransAB", fw_a, fw_b, "A->B", "B->A")
        trans_bc = _make_transformer("_TransBC", fw_b, fw_c, "B->C", "C->B")
        registry = _registry_with(trans_ab, trans_bc)

        chain = registry.get_transformation_chain(fw_a, fw_c)

        assert chain is not None, (
            "get_transformation_chain returned None: it must compose registered edges "
            "(A -> B -> C) instead of only chaining through pa.Table"
        )
        assert chain == [trans_ab, trans_bc]

    def test_three_hop_composition_resolves(self) -> None:
        """Only A <-> B, B <-> C, C <-> D exist: (A, D) resolves to the 3-hop chain
        [T(A,B), T(B,C), T(C,D)].
        """
        fw_a = _new_framework("_FwA")
        fw_b = _new_framework("_FwB")
        fw_c = _new_framework("_FwC")
        fw_d = _new_framework("_FwD")
        trans_ab = _make_transformer("_TransAB", fw_a, fw_b, "A->B", "B->A")
        trans_bc = _make_transformer("_TransBC", fw_b, fw_c, "B->C", "C->B")
        trans_cd = _make_transformer("_TransCD", fw_c, fw_d, "C->D", "D->C")
        registry = _registry_with(trans_ab, trans_bc, trans_cd)

        chain = registry.get_transformation_chain(fw_a, fw_d)

        assert chain is not None, (
            "get_transformation_chain returned None: it must compose the registered 3-hop path A -> B -> C -> D"
        )
        assert chain == [trans_ab, trans_bc, trans_cd]

    def test_composed_chain_is_executable_by_apply_chain(self) -> None:
        """The composed 2-hop chain runs end to end via ``apply_chain``."""
        fw_a = _new_framework("_FwA")
        fw_b = _new_framework("_FwB")
        fw_c = _new_framework("_FwC")
        trans_ab = _make_transformer("_TransAB", fw_a, fw_b, "A->B", "B->A")
        trans_bc = _make_transformer("_TransBC", fw_b, fw_c, "B->C", "C->B")
        registry = _registry_with(trans_ab, trans_bc)

        chain = registry.get_transformation_chain(fw_a, fw_c)

        assert chain is not None, (
            "get_transformation_chain returned None: composed chains must exist so apply_chain can execute them"
        )
        result = registry.apply_chain(fw_a, fw_c, chain, ["start"], None)
        assert result == ["start", "A->B", "B->C"]


class TestPathSelection:
    """Direct edge beats multi-hop; shortest path beats longer paths."""

    def test_direct_edge_wins_over_two_hop_path(self) -> None:
        """With a direct A <-> C transformer AND a 2-hop A -> B -> C path registered,
        the direct single-element chain is returned (direct lookup happens first).
        """
        fw_a = _new_framework("_FwA")
        fw_b = _new_framework("_FwB")
        fw_c = _new_framework("_FwC")
        trans_ab = _make_transformer("_TransAB", fw_a, fw_b, "A->B", "B->A")
        trans_bc = _make_transformer("_TransBC", fw_b, fw_c, "B->C", "C->B")
        trans_ac = _make_transformer("_TransAC", fw_a, fw_c, "A->C", "C->A")
        registry = _registry_with(trans_ab, trans_bc, trans_ac)

        chain = registry.get_transformation_chain(fw_a, fw_c)

        assert chain == [trans_ac]

    def test_shortest_path_wins_over_longer_path(self) -> None:
        """With a 3-hop path A -> B1 -> B2 -> C registered FIRST and a 2-hop path
        A -> X -> C registered second, the 2-hop chain is returned.

        Registering the longer path first makes this discriminating: a naive
        depth-first walk in insertion order would find the 3-hop path, breadth-first
        shortest path finds the 2-hop one.
        """
        fw_a = _new_framework("_FwA")
        fw_b1 = _new_framework("_FwB1")
        fw_b2 = _new_framework("_FwB2")
        fw_x = _new_framework("_FwX")
        fw_c = _new_framework("_FwC")
        trans_ab1 = _make_transformer("_TransAB1", fw_a, fw_b1, "A->B1", "B1->A")
        trans_b1b2 = _make_transformer("_TransB1B2", fw_b1, fw_b2, "B1->B2", "B2->B1")
        trans_b2c = _make_transformer("_TransB2C", fw_b2, fw_c, "B2->C", "C->B2")
        trans_ax = _make_transformer("_TransAX", fw_a, fw_x, "A->X", "X->A")
        trans_xc = _make_transformer("_TransXC", fw_x, fw_c, "X->C", "C->X")
        registry = _registry_with(trans_ab1, trans_b1b2, trans_b2c, trans_ax, trans_xc)

        chain = registry.get_transformation_chain(fw_a, fw_c)

        assert chain is not None, (
            "get_transformation_chain returned None: it must compose registered edges "
            "and pick the shortest path A -> X -> C"
        )
        assert chain == [trans_ax, trans_xc], (
            f"Expected the SHORTEST path [_TransAX, _TransXC], got {[t.__name__ for t in chain]}"
        )


class TestDeterministicTieBreak:
    """Equal-length paths: first-registered edge order (map insertion order) wins."""

    def test_tie_break_is_first_registered_edge_order(self) -> None:
        """Two distinct 2-hop paths A -> B1 -> C and A -> B2 -> C exist, with the B1
        edges registered first: the chain through B1 is returned.
        """
        fw_a = _new_framework("_FwA")
        fw_b1 = _new_framework("_FwB1")
        fw_b2 = _new_framework("_FwB2")
        fw_c = _new_framework("_FwC")
        trans_ab1 = _make_transformer("_TransAB1", fw_a, fw_b1, "A->B1", "B1->A")
        trans_b1c = _make_transformer("_TransB1C", fw_b1, fw_c, "B1->C", "C->B1")
        trans_ab2 = _make_transformer("_TransAB2", fw_a, fw_b2, "A->B2", "B2->A")
        trans_b2c = _make_transformer("_TransB2C", fw_b2, fw_c, "B2->C", "C->B2")
        registry = _registry_with(trans_ab1, trans_b1c, trans_ab2, trans_b2c)

        chain = registry.get_transformation_chain(fw_a, fw_c)

        assert chain is not None, (
            "get_transformation_chain returned None: it must compose one of the two registered 2-hop paths"
        )
        assert chain == [trans_ab1, trans_b1c], (
            "Tie-break must be first-registered edge order (insertion order of "
            f"transformer_map), i.e. the B1 path; got {[t.__name__ for t in chain]}"
        )

    def test_repeated_calls_return_the_same_chain(self) -> None:
        """With two equally short paths registered, repeated calls in one process
        return the SAME chain object list.
        """
        fw_a = _new_framework("_FwA")
        fw_b1 = _new_framework("_FwB1")
        fw_b2 = _new_framework("_FwB2")
        fw_c = _new_framework("_FwC")
        trans_ab1 = _make_transformer("_TransAB1", fw_a, fw_b1, "A->B1", "B1->A")
        trans_b1c = _make_transformer("_TransB1C", fw_b1, fw_c, "B1->C", "C->B1")
        trans_ab2 = _make_transformer("_TransAB2", fw_a, fw_b2, "A->B2", "B2->A")
        trans_b2c = _make_transformer("_TransB2C", fw_b2, fw_c, "B2->C", "C->B2")
        registry = _registry_with(trans_ab1, trans_b1c, trans_ab2, trans_b2c)

        first = registry.get_transformation_chain(fw_a, fw_c)
        second = registry.get_transformation_chain(fw_a, fw_c)
        third = registry.get_transformation_chain(fw_a, fw_c)

        assert first is not None, "get_transformation_chain returned None: composition is not implemented"
        assert first == second == third, (
            f"Chain resolution must be deterministic across calls; got {first}, {second}, {third}"
        )


class TestExistingBehaviorGuards:
    """Pins that already pass today and must survive the BFS rewrite."""

    def test_no_path_returns_none_for_disconnected_framework(self) -> None:
        """Only A <-> B is registered; C is disconnected: (A, C) resolves to None."""
        fw_a = _new_framework("_FwA")
        fw_b = _new_framework("_FwB")
        fw_c = _new_framework("_FwC")
        trans_ab = _make_transformer("_TransAB", fw_a, fw_b, "A->B", "B->A")
        registry = _registry_with(trans_ab)

        assert registry.get_transformation_chain(fw_a, fw_c) is None

    def test_real_registry_resolves_pandas_polars_via_pyarrow_hub(self) -> None:
        """The default registry still resolves pandas <-> polars as the exact 2-hop
        pa.Table chain [T(pandas, pa.Table), T(pa.Table, polars)]."""
        pd = pytest.importorskip("pandas")
        pl = pytest.importorskip("polars")
        pa = pytest.importorskip("pyarrow")

        registry = ComputeFrameworkTransformer()

        chain = registry.get_transformation_chain(pd.DataFrame, pl.DataFrame)

        assert chain is not None
        assert len(chain) == 2
        assert chain == [
            registry.transformer_map[(pd.DataFrame, pa.Table)],
            registry.transformer_map[(pa.Table, pl.DataFrame)],
        ]
