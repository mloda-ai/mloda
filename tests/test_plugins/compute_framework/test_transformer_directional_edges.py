"""``ComputeFrameworkTransformer.add`` registers each edge only when the matching
transform direction is actually overridden, and rejects a reverse-edge collision the
same way it rejects a forward-edge collision.

Contract:

  * Requirement A1: the FORWARD edge ``(framework(), other_framework())`` is registered
    only when ``transform_fw_to_other_fw`` is overridden relative to ``BaseTransformer``
    (symmetric with the reverse guard). A reverse-only transformer registers ONLY the
    reverse edge.
  * Requirement A2: ``get_transformation_chain`` never routes through the absent forward
    edge of a reverse-only transformer.
  * Requirement A3: a transformer overriding NEITHER direction registers no edges and
    ``add()`` returns False.
  * Requirement A4 (guard): two-way transformers register both edges; forward-only
    transformers register only the forward edge.
  * Requirement B1: registering a transformer whose REVERSE edge is already claimed by a
    DIFFERENT transformer class raises ``ValueError``.
  * Requirement B2 (guard): re-adding the exact same transformer class is idempotent,
    returns True and does not raise, for both forward-only and reverse-only transformers.

Isolation note (mirrors ``test_one_way_transformer_edges.py``): every synthetic
transformer and its framework types are built fresh inside a factory per test, and each
registry under test has its map reset to ``{}`` so only the test's own ``add()`` calls are
reflected. Fresh framework types are unreachable from any real framework; the colliding
pair in B1 lives entirely on function-local types that are dropped once the test returns,
so it can never poison another test's global transformer discovery.
"""

from __future__ import annotations

import gc
import logging
import weakref
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


def _make_forward_only_transformer(name: str, fw: type[Any], other_fw: type[Any]) -> type[BaseTransformer]:
    """Overrides ONLY ``transform_fw_to_other_fw``; reverse hook stays the raising default."""

    class _ForwardOnly(BaseTransformer):
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

    _ForwardOnly.__name__ = name
    _ForwardOnly.__qualname__ = name
    return _ForwardOnly


def _make_reverse_only_transformer(name: str, fw: type[Any], other_fw: type[Any]) -> type[BaseTransformer]:
    """Overrides ONLY ``transform_other_fw_to_fw``; forward hook stays the raising default.

    Such a transformer must register ONLY the reverse edge ``(other_framework, framework)``.
    """

    class _ReverseOnly(BaseTransformer):
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
        def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
            return [*data, f"{other_fw.__name__}->{fw.__name__}"]

    _ReverseOnly.__name__ = name
    _ReverseOnly.__qualname__ = name
    return _ReverseOnly


def _make_two_way_transformer(name: str, fw: type[Any], other_fw: type[Any]) -> type[BaseTransformer]:
    """Overrides BOTH directions; registers both edges."""

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


class _SeedSentinelTransformer(BaseTransformer):
    """Module-level occupant used only to pre-seed an already-claimed edge in the B1 test.

    It is a concrete transformer (so it satisfies ``type[BaseTransformer]`` in the map), but
    its import hooks RAISE, so ``check_imports`` is False and global transformer discovery's
    ``add()`` returns False without ever registering it. The B1 collision therefore stays
    confined to the test's local registry (where it is seeded directly, bypassing add()) and
    can never poison another test's global discovery. Only one genuinely-registering
    throwaway transformer (``_ReverseOnlyOF``) participates, and alone it collides with
    nothing globally.
    """

    @classmethod
    def framework(cls) -> Any:
        return type("_SentinelLeft", (), {})

    @classmethod
    def other_framework(cls) -> Any:
        return type("_SentinelRight", (), {})

    @classmethod
    def import_fw(cls) -> None:
        raise ImportError

    @classmethod
    def import_other_fw(cls) -> None:
        raise ImportError

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        return data

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
        return data


def _make_no_direction_transformer(name: str, fw: type[Any], other_fw: type[Any]) -> type[BaseTransformer]:
    """Overrides NEITHER transform direction; both hooks stay the raising default.

    Import hooks succeed so ``check_imports`` passes and ``add()`` reaches its edge-
    registration logic; with no direction implemented it must register nothing.
    """

    class _NoDirection(BaseTransformer):
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

    _NoDirection.__name__ = name
    _NoDirection.__qualname__ = name
    return _NoDirection


class TestForwardEdgeGuard:
    def test_reverse_only_transformer_registers_only_the_reverse_edge(self) -> None:
        """(A1) A reverse-only transformer registers ``(other, framework)`` and NOT
        ``(framework, other)``.
        """
        fw = type("_FwLeft", (), {})
        other_fw = type("_FwRight", (), {})
        reverse_only = _make_reverse_only_transformer("_ReverseOnlyLR", fw, other_fw)
        registry = _empty_registry()

        assert registry.add(reverse_only) is True

        assert registry.transformer_map.get((other_fw, fw)) is reverse_only
        assert (fw, other_fw) not in registry.transformer_map, (
            "add() must not register the forward edge of a reverse-only transformer: its "
            "transform_fw_to_other_fw is BaseTransformer's raising default"
        )

    def test_chain_never_routes_through_absent_forward_edge(self) -> None:
        """(A2) The only conceivable route X -> Z goes X -> Y via the ABSENT forward edge of
        a reverse-only transformer T(framework=X, other=Y), then Y -> Z via a two-way
        transformer. The forward edge is unregistered, so the chain must be None.
        """
        fw_x = type("_FwX", (), {})
        fw_y = type("_FwY", (), {})
        fw_z = type("_FwZ", (), {})
        reverse_only_xy = _make_reverse_only_transformer("_ReverseOnlyXY", fw_x, fw_y)
        two_way_yz = _make_two_way_transformer("_TwoWayYZ", fw_y, fw_z)

        registry = _empty_registry()
        assert registry.add(reverse_only_xy) is True
        assert registry.add(two_way_yz) is True

        chain = registry.get_transformation_chain(fw_x, fw_z)

        assert chain is None, (
            "get_transformation_chain must not route through the absent forward edge of a "
            f"reverse-only transformer; got {None if chain is None else [t.__name__ for t in chain]}"
        )

    def test_no_direction_transformer_registers_nothing_and_returns_false(self) -> None:
        """(A3) A transformer overriding neither direction registers no edges; add() is False."""
        fw = type("_FwNoneA", (), {})
        other_fw = type("_FwNoneB", (), {})
        no_direction = _make_no_direction_transformer("_NoDirectionAB", fw, other_fw)
        registry = _empty_registry()

        assert registry.add(no_direction) is False
        assert (fw, other_fw) not in registry.transformer_map
        assert (other_fw, fw) not in registry.transformer_map


class TestExistingEdgeBehaviorPreserved:
    def test_two_way_transformer_registers_both_edges(self) -> None:
        """(A4) A two-way transformer keeps registering both directions."""
        fw = type("_FwTwoA", (), {})
        other_fw = type("_FwTwoB", (), {})
        two_way = _make_two_way_transformer("_TwoWayAB", fw, other_fw)
        registry = _empty_registry()

        assert registry.add(two_way) is True
        assert registry.transformer_map.get((fw, other_fw)) is two_way
        assert registry.transformer_map.get((other_fw, fw)) is two_way

    def test_forward_only_transformer_registers_only_forward_edge(self) -> None:
        """(A4) A forward-only transformer registers only ``(framework, other)``."""
        fw = type("_FwFwdA", (), {})
        other_fw = type("_FwFwdB", (), {})
        forward_only = _make_forward_only_transformer("_ForwardOnlyAB", fw, other_fw)
        registry = _empty_registry()

        assert registry.add(forward_only) is True
        assert registry.transformer_map.get((fw, other_fw)) is forward_only
        assert (other_fw, fw) not in registry.transformer_map


class TestReverseEdgeConflict:
    def test_reverse_edge_collision_with_different_class_raises(self) -> None:
        """(B1) The edge (F, O) is already claimed by a DIFFERENT transformer class. A
        reverse-only transformer T(framework=O, other=F) whose REVERSE edge is exactly (F, O)
        then tries to claim it; add() must raise ValueError and leave the incumbent in place.

        The incumbent is seeded directly into the local map so the deliberate collision never
        reaches global transformer discovery (see the module-level isolation note).
        """
        fw_f = type("_FwF", (), {})
        fw_o = type("_FwO", (), {})
        reverse_only = _make_reverse_only_transformer("_ReverseOnlyOF", fw_o, fw_f)

        registry = _empty_registry()
        registry.transformer_map[(fw_f, fw_o)] = _SeedSentinelTransformer

        with pytest.raises(ValueError) as excinfo:
            registry.add(reverse_only)

        assert "already registered" in str(excinfo.value), str(excinfo.value)
        assert registry.transformer_map.get((fw_f, fw_o)) is _SeedSentinelTransformer, (
            "the conflicting reverse edge must not overwrite the incumbent transformer"
        )

    def test_forward_only_reregistration_is_idempotent(self) -> None:
        """(B2) Re-adding the same forward-only transformer returns True and does not raise."""
        fw = type("_FwIdemFwdA", (), {})
        other_fw = type("_FwIdemFwdB", (), {})
        forward_only = _make_forward_only_transformer("_ForwardOnlyIdem", fw, other_fw)
        registry = _empty_registry()

        assert registry.add(forward_only) is True
        assert registry.add(forward_only) is True
        assert registry.transformer_map.get((fw, other_fw)) is forward_only

    def test_reverse_only_reregistration_is_idempotent(self) -> None:
        """(B2) Re-adding the same reverse-only transformer returns True and does not raise."""
        fw = type("_FwIdemRevA", (), {})
        other_fw = type("_FwIdemRevB", (), {})
        reverse_only = _make_reverse_only_transformer("_ReverseOnlyIdem", fw, other_fw)
        registry = _empty_registry()

        assert registry.add(reverse_only) is True
        assert registry.add(reverse_only) is True
        assert registry.transformer_map.get((other_fw, fw)) is reverse_only


_CFW_TRANSFORMER_LOGGER = "mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer"


def _cfw_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """WARNING records emitted by the cfw_transformer module logger."""
    return [
        record
        for record in caplog.records
        if record.name == _CFW_TRANSFORMER_LOGGER and record.levelno == logging.WARNING
    ]


class TestZeroEdgeWarning:
    """``add()`` WARNS when imports pass but neither transform direction is overridden
    (zero edges registered), because that silence hides an authoring bug such as a
    typo'd override. Failed imports and edge-registering transformers stay silent.
    """

    def test_no_direction_transformer_warns_naming_the_class(self, caplog: pytest.LogCaptureFixture) -> None:
        """Passing imports plus zero overridden directions: add() is False, no edges, and
        exactly one WARNING from the cfw_transformer module logger names the class.

        The registry is built BEFORE the transformer class exists, so the registry
        constructor's discovery pass cannot have already add()-ed (and warned about) the
        class; the explicit ``registry.add(...)`` below is the sole possible warning source.
        """
        registry = _empty_registry()
        fw = type("_FwWarnA", (), {})
        other_fw = type("_FwWarnB", (), {})
        no_direction = _make_no_direction_transformer("_NoDirectionWarnAB", fw, other_fw)

        with caplog.at_level(logging.WARNING, logger=_CFW_TRANSFORMER_LOGGER):
            assert registry.add(no_direction) is False

        assert (fw, other_fw) not in registry.transformer_map
        assert (other_fw, fw) not in registry.transformer_map

        warnings = _cfw_warnings(caplog)
        assert len(warnings) == 1, (
            "add() must emit exactly one WARNING via the cfw_transformer module logger when a "
            f"transformer with passing imports registers zero edges; got {len(warnings)} records"
        )
        assert "_NoDirectionWarnAB" in warnings[0].getMessage(), (
            "the warning message must name the transformer class so the author can find the "
            f"typo'd override; got: {warnings[0].getMessage()!r}"
        )

    def test_failed_imports_do_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """A transformer whose import hooks raise is a normal absent-dependency case:
        add() is False and NO warning is emitted."""

        class _ImportlessLocal(BaseTransformer):
            @classmethod
            def framework(cls) -> Any:
                return type("_FwImportlessA", (), {})

            @classmethod
            def other_framework(cls) -> Any:
                return type("_FwImportlessB", (), {})

            @classmethod
            def import_fw(cls) -> None:
                raise ImportError

            @classmethod
            def import_other_fw(cls) -> None:
                raise ImportError

            @classmethod
            def transform_fw_to_other_fw(cls, data: Any) -> Any:
                return data

            @classmethod
            def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
                return data

        registry = _empty_registry()

        with caplog.at_level(logging.WARNING, logger=_CFW_TRANSFORMER_LOGGER):
            assert registry.add(_ImportlessLocal) is False

        assert _cfw_warnings(caplog) == [], "failed check_imports() must stay silent: it is not an authoring bug"

    def test_edge_registering_transformer_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """A transformer registering at least one edge succeeds silently."""
        fw = type("_FwQuietA", (), {})
        other_fw = type("_FwQuietB", (), {})
        two_way = _make_two_way_transformer("_TwoWayQuietAB", fw, other_fw)
        registry = _empty_registry()

        with caplog.at_level(logging.WARNING, logger=_CFW_TRANSFORMER_LOGGER):
            assert registry.add(two_way) is True

        assert _cfw_warnings(caplog) == [], "a transformer that registers edges must not warn"

    def test_zero_edge_dedupe_does_not_retain_transformer_class(self) -> None:
        """The zero-edge warning dedupe does not keep the transformer class alive.

        After the only strong reference to a warned-about class is dropped, the class must
        be collectable. ``BaseTransformer.__subclasses__`` holds only weak references, so
        nothing else pins it; only a strong dedupe container (e.g. a plain set) would.
        """
        registry = _empty_registry()
        fw = type("_FwWeakRefA", (), {})
        other_fw = type("_FwWeakRefB", (), {})
        transformer_cls = _make_no_direction_transformer("_NoDirectionWeakRef", fw, other_fw)

        assert registry.add(transformer_cls) is False

        ref = weakref.ref(transformer_cls)
        del transformer_cls
        gc.collect()

        assert ref() is None, (
            "the zero-edge warning dedupe must not retain transformer classes for the process "
            "lifetime; BaseTransformer.__subclasses__ is weak, so nothing else pins the class"
        )
