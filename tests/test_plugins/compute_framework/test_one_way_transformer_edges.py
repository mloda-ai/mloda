"""One-way transformers register ONLY their forward edge.

For a one-way transformer (e.g. the FileSource materializers, whose reverse direction is
meaningless), a registered reverse edge would be a trap: the BFS in
``get_transformation_chain`` could route a chain through it, and ``apply_chain`` would then
crash mid-chain with a bare ``NotImplementedError`` from ``transform_other_fw_to_fw``.

Contract:

  * ``add()`` detects STRUCTURALLY that a transformer does not override
    ``transform_other_fw_to_fw`` relative to ``BaseTransformer`` (function identity via the
    ``_underlying`` idiom of ``BaseInputData._is_overridden``) and then registers ONLY the
    forward edge ``(framework(), other_framework())``, never the reverse pair.
  * The two FileSource transformers do not override ``transform_other_fw_to_fw``, so the
    default registry contains ``(FileSource, dict)`` and ``(FileSource, pa.Table)`` but NOT
    the reverse pairs.
  * With the reverse edge unregistered, a route that would only exist via a one-way
    transformer's reverse edge resolves to ``None`` instead of a chain that would crash.

ANTI-PATTERN (pinned by ``TestRaisingOverrideAntiPatternRegistersTheEdge`` below): declaring
"this direction is not supported" by OVERRIDING the unused hook with a bare
``raise NotImplementedError``. Since ``add()`` decides on function identity, ANY override, even
a purely raising one, is the "register this edge" signal: the dead edge lands in the map, the
BFS routes a chain through it, and ``apply_chain`` crashes mid-chain with a bare
``NotImplementedError`` instead of the caller getting a clean "no chain exists" ``None``.
Declare a direction unsupported by leaving its hook unimplemented, never by raising.

Isolation note: the synthetic frameworks and transformers are built fresh inside factory
functions per test. Their import hooks succeed (``add()`` must accept them), so a lingering
function-local class can be picked up by another test's ``ComputeFrameworkTransformer()``
discovery; that is harmless because each call creates FRESH framework types that are
unreachable from any real framework and can never collide on a pair.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
    ComputeFrameworkTransformer,
)


def _empty_registry() -> ComputeFrameworkTransformer:
    """A registry whose map only reflects the add() calls made by the test itself."""
    registry = ComputeFrameworkTransformer()
    registry.transformer_map = {}
    return registry


def _make_one_way_transformer(name: str, fw: type[Any], other_fw: type[Any]) -> type[BaseTransformer]:
    """A transformer overriding ONLY transform_fw_to_other_fw; the reverse hook stays
    BaseTransformer's raising default. Import hooks succeed so add() accepts it."""

    class _OneWay(BaseTransformer):
        @classmethod
        def framework(cls) -> Any:
            return fw

        @classmethod
        def other_framework(cls) -> Any:
            return other_fw

        @classmethod
        def import_fw(cls) -> None:
            pass

        @classmethod
        def import_other_fw(cls) -> None:
            pass

        @classmethod
        def transform_fw_to_other_fw(cls, data: Any) -> Any:
            return [*data, f"{fw.__name__}->{other_fw.__name__}"]

    _OneWay.__name__ = name
    _OneWay.__qualname__ = name
    return _OneWay


def _make_two_way_transformer(name: str, fw: type[Any], other_fw: type[Any]) -> type[BaseTransformer]:
    """A transformer overriding BOTH directions. Import hooks succeed so add() accepts it."""

    class _TwoWay(BaseTransformer):
        @classmethod
        def framework(cls) -> Any:
            return fw

        @classmethod
        def other_framework(cls) -> Any:
            return other_fw

        @classmethod
        def import_fw(cls) -> None:
            pass

        @classmethod
        def import_other_fw(cls) -> None:
            pass

        @classmethod
        def transform_fw_to_other_fw(cls, data: Any) -> Any:
            return [*data, f"{fw.__name__}->{other_fw.__name__}"]

        @classmethod
        def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
            return [*data, f"{other_fw.__name__}->{fw.__name__}"]

    _TwoWay.__name__ = name
    _TwoWay.__qualname__ = name
    return _TwoWay


def _make_raising_override_transformer(name: str, fw: type[Any], other_fw: type[Any]) -> type[BaseTransformer]:
    """ANTI-PATTERN factory: spells the unused direction as a bare raising override instead of
    omitting it. Contrast ``_make_one_way_transformer``, which omits the hook."""

    class _RaisingOverride(BaseTransformer):
        @classmethod
        def framework(cls) -> Any:
            return fw

        @classmethod
        def other_framework(cls) -> Any:
            return other_fw

        @classmethod
        def import_fw(cls) -> None:
            pass

        @classmethod
        def import_other_fw(cls) -> None:
            pass

        @classmethod
        def transform_fw_to_other_fw(cls, data: Any) -> Any:
            return [*data, f"{fw.__name__}->{other_fw.__name__}"]

        @classmethod
        def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
            raise NotImplementedError

    _RaisingOverride.__name__ = name
    _RaisingOverride.__qualname__ = name
    return _RaisingOverride


class TestAddRegistersOneWayTransformersForwardOnly:
    def test_one_way_transformer_registers_only_the_forward_edge(self) -> None:
        """(1a) A transformer WITHOUT a transform_other_fw_to_fw override registers only
        (framework(), other_framework()).
        """
        fw_src = type("_FwSrc", (), {})
        fw_dst = type("_FwDst", (), {})
        one_way = _make_one_way_transformer("_OneWaySrcDst", fw_src, fw_dst)
        registry = _empty_registry()

        assert registry.add(one_way) is True

        assert registry.transformer_map.get((fw_src, fw_dst)) is one_way
        assert (fw_dst, fw_src) not in registry.transformer_map, (
            "add() must not register the reverse edge of a one-way transformer: its "
            "transform_other_fw_to_fw is BaseTransformer's raising default, so the edge "
            "can only crash apply_chain with a bare NotImplementedError"
        )

    def test_two_way_transformer_registers_both_edges(self) -> None:
        """(1b) Guard: a transformer overriding BOTH directions still registers both pairs."""
        fw_src = type("_FwSrc", (), {})
        fw_dst = type("_FwDst", (), {})
        two_way = _make_two_way_transformer("_TwoWaySrcDst", fw_src, fw_dst)
        registry = _empty_registry()

        assert registry.add(two_way) is True

        assert registry.transformer_map.get((fw_src, fw_dst)) is two_way
        assert registry.transformer_map.get((fw_dst, fw_src)) is two_way


class TestRealRegistryFileSourceEdgesAreForwardOnly:
    def test_file_source_dict_edge_is_forward_only(self) -> None:
        """(1c) Default registry: (FileSource, dict) is registered, (dict, FileSource) is not."""
        from mloda.core.abstract_plugins.components.input_data.file_source import FileSource

        registry = ComputeFrameworkTransformer()

        assert (FileSource, dict) in registry.transformer_map
        assert (dict, FileSource) not in registry.transformer_map, (
            "dict -> FileSource is meaningless; a registered reverse edge lets the BFS "
            "route chains through it and crash with NotImplementedError"
        )

    def test_file_source_pyarrow_edge_is_forward_only(self) -> None:
        """(1c) Default registry: (FileSource, pa.Table) is registered, (pa.Table, FileSource) is not."""
        pa = pytest.importorskip("pyarrow")
        from mloda.core.abstract_plugins.components.input_data.file_source import FileSource

        registry = ComputeFrameworkTransformer()

        assert (FileSource, pa.Table) in registry.transformer_map
        assert (pa.Table, FileSource) not in registry.transformer_map, (
            "pa.Table -> FileSource is meaningless; a registered reverse edge lets the BFS "
            "route chains through it and crash with NotImplementedError"
        )


class TestBfsNeverRoutesThroughOneWayReverseEdges:
    def test_route_only_reachable_via_one_way_reverse_edge_resolves_to_none(self) -> None:
        """(1d) The ONLY conceivable route from A to C is A -> B via the REVERSE edge of the
        one-way transformer T(B, A), then B -> C via a two-way transformer. That route would
        crash apply_chain (T's transform_other_fw_to_fw is the raising base default), so
        get_transformation_chain(A, C) must return None.
        """
        fw_a = type("_FwA", (), {})
        fw_b = type("_FwB", (), {})
        fw_c = type("_FwC", (), {})
        one_way_b_to_a = _make_one_way_transformer("_OneWayBToA", fw_b, fw_a)
        two_way_bc = _make_two_way_transformer("_TwoWayBC", fw_b, fw_c)

        registry = _empty_registry()
        assert registry.add(one_way_b_to_a) is True
        assert registry.add(two_way_bc) is True

        chain = registry.get_transformation_chain(fw_a, fw_c)

        assert chain is None, (
            "get_transformation_chain must not build a chain through the reverse edge of a "
            f"one-way transformer; got {[t.__name__ for t in chain]} which would crash "
            "apply_chain with a bare NotImplementedError"
        )


class TestRaisingOverrideAntiPatternRegistersTheEdge:
    """Pins the ANTI-PATTERN described in the module docstring."""

    def test_raising_override_registers_the_reverse_edge_but_omission_does_not(self) -> None:
        """(2a) Same intent, two spellings: a raising override registers the reverse edge,
        omitting the hook does not.

        Each transformer gets its OWN fresh framework pair, never a shared one: two classes
        claiming the same pair collide in the next registry's discovery pass and poison
        unrelated tests.
        """
        anti_src = type("_FwAntiSrc", (), {})
        anti_dst = type("_FwAntiDst", (), {})
        ok_src = type("_FwOmittedSrc", (), {})
        ok_dst = type("_FwOmittedDst", (), {})
        anti_pattern = _make_raising_override_transformer("_RaisingOverrideSrcDst", anti_src, anti_dst)
        correct = _make_one_way_transformer("_OmittedSrcDst", ok_src, ok_dst)

        registry = _empty_registry()
        assert registry.add(anti_pattern) is True
        assert registry.add(correct) is True

        assert registry.transformer_map.get((anti_dst, anti_src)) is anti_pattern, (
            "a raising override still gives transform_other_fw_to_fw a new function identity, "
            "which IS add()'s 'register this edge' signal, so the dead reverse edge is registered"
        )
        assert (ok_dst, ok_src) not in registry.transformer_map, (
            "omitting transform_other_fw_to_fw leaves BaseTransformer's default in place, so no "
            "reverse edge is registered"
        )

        assert registry.transformer_map.get((anti_src, anti_dst)) is anti_pattern
        assert registry.transformer_map.get((ok_src, ok_dst)) is correct

    def test_chain_routes_through_the_raising_edge_and_apply_chain_raises(self) -> None:
        """(2b) The failure mode. Topology mirrors (1d) with the anti-pattern transformer in
        place of the correct one-way: the only route A -> C runs through the raising REVERSE
        edge of T(framework=B, other=A). Instead of an honest None, the BFS hands back a chain
        that looks valid and apply_chain dies inside the raising hook.
        """
        fw_a = type("_FwAntiA", (), {})
        fw_b = type("_FwAntiB", (), {})
        fw_c = type("_FwAntiC", (), {})
        anti_pattern_b_to_a = _make_raising_override_transformer("_RaisingOverrideBToA", fw_b, fw_a)
        two_way_bc = _make_two_way_transformer("_TwoWayAntiBC", fw_b, fw_c)

        registry = _empty_registry()
        assert registry.add(anti_pattern_b_to_a) is True
        assert registry.add(two_way_bc) is True

        chain = registry.get_transformation_chain(fw_a, fw_c)

        assert chain == [anti_pattern_b_to_a, two_way_bc], (
            "the registered raising edge makes A -> B -> C look routable, so the BFS selects a "
            f"chain instead of returning None; got {None if chain is None else [t.__name__ for t in chain]}"
        )

        with pytest.raises(NotImplementedError):
            registry.apply_chain(fw_a, fw_c, chain, ["seed"], None)
